#!/usr/bin/env python3
"""
engine.py — Headless Strategy (MicroStrategy) + market tracker.

Fetches market/crypto data and regenerates a daily Markdown report plus a
rolling front page. Designed to run unattended in GitHub Actions and never
crash the workflow: every data source is wrapped so that a single failure
degrades one section (shows "N/A") instead of aborting the whole run.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

try:
    import yfinance as yf
except Exception as exc:
    raise SystemExit(f"yfinance failed to import — is it installed? ({exc})")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BTC_FUTURES_TICKER = "BTC=F"            
MSTR_GRID_TICKERS = ["MSTR", "MSTY", "STRC", "STRK", "STRF", "STRD"]

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_TREASURY_PATH = "/companies/public_treasury/bitcoin"
STRATEGY_CG_SYMBOL = "MSTR.US"          
STRATEGY_CG_NAME_HINTS = ("strategy", "microstrategy")

FMP_BASE = "https://financialmodelingprep.com/api/v3"
PENNY_MAX_PRICE = 5.0
PENNY_MIN_MARKET_CAP = 100_000_000      
PENNY_UNIVERSE_CAP = 1000               
SCREENER_TOP_N = 20                     
QUOTE_CHUNK = 50                        

FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()
CG_DEMO_API_KEY = os.environ.get("CG_DEMO_API_KEY", "").strip()  

REPORTS_DIR = Path("reports")
INDEX_FILE = Path("index.md")
ROLLING_DAYS = 30
HTTP_TIMEOUT = 20
RETRY_ATTEMPTS = 3

NOW_UTC = datetime.now(timezone.utc)
DATE_STR = NOW_UTC.date().isoformat()   
TIMESTAMP_STR = NOW_UTC.strftime("%Y-%m-%d %H:%M UTC")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tracker")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _retry(fn, *, attempts: int = RETRY_ATTEMPTS, base_delay: float = 2.0,
           label: str = "request"):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            log.warning("%s attempt %d/%d failed: %s", label, i, attempts, exc)
            if i < attempts:
                time.sleep(base_delay * i)
    log.error("%s: giving up after %d attempts", label, attempts)
    return None


def fmt_money(x: Optional[float]) -> str:
    return f"${x:,.2f}" if isinstance(x, (int, float)) else "N/A"


def fmt_pct(x: Optional[float]) -> str:
    return f"{x:+.2f}%" if isinstance(x, (int, float)) else "N/A"


def fmt_compact(x: Optional[float]) -> str:
    if not isinstance(x, (int, float)):
        return "N/A"
    a = abs(x)
    if a >= 1e12:
        return f"${x / 1e12:,.2f}T"
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.1f}M"
    return f"${x:,.0f}"


def _coerce_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            return float(m.group())
    return None


# --------------------------------------------------------------------------- #
# yfinance & FMP Quotes
# --------------------------------------------------------------------------- #

def fetch_quote_yf(ticker: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    def _do():
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            raise ValueError("empty history")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise ValueError("no closes")
        
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
        pct = ((last / prev) - 1.0) * 100.0 if prev else None
        
        ath = float(closes.max())
        ath_pct = ((last / ath) - 1.0) * 100.0 if ath > 0 else None
        
        return last, pct, ath_pct

    result = _retry(_do, label=f"yfinance:{ticker}")
    return result if result is not None else (None, None, None)


def get_btc_futures() -> Optional[float]:
    price, _, _ = fetch_quote_yf(BTC_FUTURES_TICKER)
    log.info("BTC=F: %s", fmt_money(price))
    return price


def get_mstr_grid() -> dict[str, tuple[Optional[float], Optional[float], Optional[float]]]:
    grid: dict[str, tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for tkr in MSTR_GRID_TICKERS:
        close, pct, ath_pct = fetch_quote_yf(tkr)
        grid[tkr] = (close, pct, ath_pct)
        time.sleep(0.4) 
    return grid

def _fmp_get(path: str, params: Optional[dict] = None):
    if not FMP_API_KEY:
        return None
    params = dict(params or {})
    params["apikey"] = FMP_API_KEY
    def _do():
        r = requests.get(f"{FMP_BASE}{path}", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code in (401, 402, 403):
            raise PermissionError(f"HTTP {r.status_code} on {path}")
        r.raise_for_status()
        return r.json()
    return _retry(_do, label=f"fmp:{path}", attempts=2)


def fmp_batch_quotes(symbols: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(symbols), QUOTE_CHUNK):
        chunk = symbols[i:i + QUOTE_CHUNK]
        data = _fmp_get(f"/quote/{','.join(chunk)}")
        if isinstance(data, list):
            for q in data:
                out.append({
                    "symbol": q.get("symbol"),
                    "name": q.get("name"),
                    "price": _coerce_float(q.get("price")),
                    "change_pct": _coerce_float(q.get("changesPercentage")),
                    "market_cap": _coerce_float(q.get("marketCap")),
                })
        time.sleep(0.3)
    return out


# --------------------------------------------------------------------------- #
# CoinGecko Treasury
# --------------------------------------------------------------------------- #

def get_treasury() -> Optional[dict]:
    def _do():
        headers = {"accept": "application/json"}
        if CG_DEMO_API_KEY:
            headers["x-cg-demo-api-key"] = CG_DEMO_API_KEY
        r = requests.get(COINGECKO_BASE + COINGECKO_TREASURY_PATH, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()

    data = _retry(_do, label="coingecko:treasury")
    if not data: return None

    companies = data.get("companies", []) or []
    record = next((c for c in companies if c.get("symbol") == STRATEGY_CG_SYMBOL), None)
    if not record:  
        record = next((c for c in companies if any(h in str(c.get("name", "")).lower() for h in STRATEGY_CG_NAME_HINTS)), None)
    if not record: return None

    holdings = _coerce_float(record.get("total_holdings"))
    entry = _coerce_float(record.get("total_entry_value_usd"))
    current = _coerce_float(record.get("total_current_value_usd"))

    pnl = pnl_pct = None
    if isinstance(current, float) and isinstance(entry, float) and entry > 0:
        pnl = current - entry
        pnl_pct = (pnl / entry) * 100.0

    return {
        "holdings": holdings,
        "entry_value": entry,
        "current_value": current,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }


# --------------------------------------------------------------------------- #
# Markdown Builders: Extra Features
# --------------------------------------------------------------------------- #

def build_grid_table(grid: dict) -> str:
    lines = ["| Ticker | Close | 24h Change | % from ATH |", "|:------:|------:|:----------:|:----------:|"]
    for tkr in MSTR_GRID_TICKERS:
        close, pct, ath_pct = grid.get(tkr, (None, None, None))
        arrow = " 🔺" if isinstance(pct, float) and pct > 0 else (" 🔻" if isinstance(pct, float) and pct < 0 else " ▪")
        ath_str = "ATH 🚀" if isinstance(ath_pct, float) and ath_pct >= 0 else fmt_pct(ath_pct)
        lines.append(f"| **{tkr}** | {fmt_money(close)} | {fmt_pct(pct)}{arrow} | {ath_str} |")
    return "\n".join(lines)


def build_mnav_section(mstr_mkt_cap: Optional[float], treasury: Optional[dict]) -> str:
    if not mstr_mkt_cap or not treasury or not treasury.get("current_value"):
        return "_mNAV data currently unavailable._\n"
    
    current_val = treasury["current_value"]
    mnav = mstr_mkt_cap / current_val if current_val > 0 else 0
    
    lines = [f"Today:  {'▓' * min(int(mnav * 10), 30)} {mnav:.2f}x (Live)"]
    
    # Generate an illustrative terminal-style history for the visual effect 
    random.seed(DATE_STR) 
    for i in range(1, 7):
        hist_mnav = mnav + random.uniform(-0.4, 0.4)
        lines.append(f"T-{i}:    {'▓' * min(max(1, int(hist_mnav * 10)), 30)} {hist_mnav:.2f}x")
        
    hist_block = "<br>\n".join(lines)
    return f"""## 📊 mNAV Premium Tracker
_MSTR Market Cap vs Treasury Value Premium_

<div style="font-family: monospace; background: #111; padding: 15px; border-radius: 8px; border: 1px solid #333; color: #F7931A; font-size: 1.1em; overflow-x: auto;">
{hist_block}
</div>
<p style="font-size: 0.85em; color: #888;">* Today's mNAV is dynamically calculated. Historical trend is illustrative simulation for layout.</p>
"""


def build_hall_of_rekt(treasury: Optional[dict]) -> str:
    mstr_loss_b = 0.0
    if treasury and isinstance(treasury.get("pnl"), float) and treasury["pnl"] < 0:
        mstr_loss_b = abs(treasury["pnl"]) / 1e9
        
    hwang_loss_b = 20.0
    dotcom_loss_b = 13.5
    
    max_loss = max(hwang_loss_b, dotcom_loss_b, mstr_loss_b, 1.0)
    
    def get_width(val):
        return min(max(int((val / max_loss) * 100), 5), 100)
        
    html = [
        "## 📉 The Hall of Rekt",
        "_Historical comparisons of unprecedented wealth evaporation._\n",
        "<div class='rekt-container'>",
        f"<div class='rekt-label'>Bill Hwang (Archegos, 2021) - $20.0B</div>",
        f"<div class='rekt-bar'><div class='rekt-fill' style='width: {get_width(hwang_loss_b)}%; background: #b30000;'>$20.0B</div></div>",
        f"<div class='rekt-label'>Old Saylor (Dot-com Peak-to-Trough, 2000) - $13.5B</div>",
        f"<div class='rekt-bar'><div class='rekt-fill' style='width: {get_width(dotcom_loss_b)}%; background: #ff8800;'>$13.5B</div></div>",
    ]
    
    if mstr_loss_b > 0:
        html.extend([
            f"<div class='rekt-label'>New Saylor (MSTR Unrealized Loss, {DATE_STR}) - ${mstr_loss_b:.2f}B</div>",
            f"<div class='rekt-bar'><div class='rekt-fill' style='width: {get_width(mstr_loss_b)}%; background: #ff4444;'>${mstr_loss_b:.2f}B</div></div>"
        ])
    else:
        html.append(f"<div class='rekt-label' style='color: #00ff00;'>New Saylor (MSTR PnL, {DATE_STR}) - IN PROFIT 🚀 No rekt here.</div>")

    html.append("</div>\n")
    return "\n".join(html)


def build_shitcoin_treasury() -> str:
    return """## 💩 Shitcoin & Distraction Treasuries
_Because not everyone has laser eyes._

| Company | Primary Distraction | Status |
|:---|:---|:---|
| **Tesla (TSLA)** | Dogecoin (DOGE) | _Elon's lingering meme addiction_ |
| **Meitu (1357.HK)** | Ethereum (ETH) | _Waiting for gas fees to drop_ |
| **Nexon (3659.T)** | Various Altcoins | _Gacha game mechanics in real life_ |
| **Reddit (RDDT)** | Polygon (MATIC) | _Avatar bagholders anonymous_ |
"""


def build_index(grid, treasury, mstr_mkt_cap) -> str:
    recent = sorted(REPORTS_DIR.glob("20*-*-*.md"), reverse=True)[:ROLLING_DAYS]
    archive_lines = [f"- [{p.stem}](reports/{p.stem}.html)" for p in recent] or ["_No reports yet._"]
    today_link = f"reports/{DATE_STR}.html"

    # Loss Summary formatting
    if treasury and isinstance(treasury.get("pnl"), float) and treasury["pnl"] < 0:
        h, entry, current, pnl = treasury["holdings"], treasury["entry_value"], treasury["current_value"], abs(treasury["pnl"])
        pnl_html = f'<div class="loss-box"><strong>Strategy holds {h:,.0f} BTC bought for {fmt_compact(entry)}; at CoinGecko\'s valuation of {fmt_compact(current)} that is an unrealized loss of <span style="color: #ff4444; font-weight: 900; font-size: 1.1em;">{fmt_compact(pnl)} ({fmt_pct(treasury["pnl_pct"])})</span></strong></div>'
    elif treasury:
        pnl_html = f'<div class="loss-box"><strong>Treasury data retrieved: Holdings {fmt_compact(treasury["current_value"])}. Currently in profit.</strong></div>'
    else:
        pnl_html = f'<div class="loss-box">_Treasury data unavailable._</div>'

    style_block = """
<style>
  html, body, .markdown-body {
    background-color: #0b0b0b !important; color: #ffffff !important; font-family: "Courier New", Courier, monospace !important;
  }
  h1, h2, h3, h4 { color: #F7931A !important; text-shadow: 0 0 12px rgba(247, 147, 26, 0.4) !important; border-bottom: none !important; }
  a { color: #F7931A !important; text-decoration: none !important; }
  a:hover { text-shadow: 0 0 8px rgba(247, 147, 26, 0.8) !important; }
  table { background-color: #111 !important; border: 1px solid #333 !important; width: 100%; max-width: 650px; }
  th { background-color: #1a1a1a !important; color: #F7931A !important; border-bottom: 2px solid #F7931A !important; padding: 10px; }
  td { color: #ffffff !important; border-bottom: 1px solid #222 !important; padding: 10px; }
  .loss-box { background-color: rgba(255, 68, 68, 0.05) !important; border-left: 4px solid #ff4444 !important; padding: 15px !important; margin: 20px 0 !important; color: #fff !important; }
  .glitch { color: #F7931A !important; font-weight: 900; }
  .rekt-container { background-color: #111 !important; border: 1px solid #333 !important; padding: 20px !important; border-radius: 8px !important; margin-bottom: 20px !important; }
  .rekt-label { color: #e0e0e0 !important; font-size: 0.9em !important; margin-bottom: 5px !important; font-weight: bold !important; }
  .rekt-bar { background: #222 !important; width: 100% !important; border-radius: 4px !important; margin-bottom: 15px !important; overflow: hidden !important; border: 1px solid #000 !important; }
  .rekt-fill { height: 24px !important; text-align: right !important; padding-right: 10px !important; color: white !important; font-weight: 900 !important; line-height: 24px !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important; }
</style>
"""

    return "\n".join([
        f"---\ntitle: \"Saylor Infinite Money Glitch Tracker\"\nlayout: null\n---\n",
        style_block,
        f"<h1><span class='glitch'>S</span>ayl<span class='glitch'>o</span>r Inf<span class='glitch'>i</span>n<span class='glitch'>i</span>t<span class='glitch'>e</span> M<span class='glitch'>o</span>n<span class='glitch'>e</span>y Gl<span class='glitch'>i</span>tch Tr<span class='glitch'>a</span>ck<span class='glitch'>e</span>r</h1>",
        f"<p><em>Last updated {TIMESTAMP_STR}</em></p>",
        f"<h3>👉 <a href=\"{today_link}\">Today's full report — {DATE_STR}</a></h3>",
        f"<h2>The S<span class='glitch'>a</span>yl<span class='glitch'>o</span>r Sc<span class='glitch'>a</span>m G<span class='glitch'>r</span>id</h2>",
        build_grid_table(grid) + "\n",
        pnl_html + "\n",
        build_hall_of_rekt(treasury),
        build_mnav_section(mstr_mkt_cap, treasury),
        build_shitcoin_treasury(),
        f"<h2>Archive — last {ROLLING_DAYS} days</h2>",
        "\n".join(archive_lines) + "\n",
        "---\n<p style=\"color: #666; font-size: 0.85em; text-align: center;\"><em>Automated. Not investment advice.</em></p>",
    ])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info("=== Saylor Tracker run: %s ===", DATE_STR)
    REPORTS_DIR.mkdir(exist_ok=True)

    grid = get_mstr_grid()
    treasury = get_treasury()
    
    # Grab MSTR market cap for the mNAV chart
    mstr_quote = fmp_batch_quotes(["MSTR"])
    mstr_mkt_cap = mstr_quote[0].get("market_cap") if mstr_quote else None

    # Overwrite index with the heavily-themed dashboard
    index_md = build_index(grid, treasury, mstr_mkt_cap)
    INDEX_FILE.write_text(index_md, encoding="utf-8")
    log.info("Wrote %s", INDEX_FILE)

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
