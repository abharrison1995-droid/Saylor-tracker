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
MSTR_GRID_TICKERS = ["MSTR", "STRC", "STRK", "STRF", "STRD"]

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
# yfinance: Fetch quotes AND calculate ATH
# --------------------------------------------------------------------------- #

def fetch_quote_yf(ticker: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (last_close, pct_change_vs_prev_close, pct_from_ath)."""
    def _do():
        # Fetching max history so we can calculate the true all-time high dynamically
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
        log.info("%-5s close=%s change=%s ath=%s", tkr, fmt_money(close), fmt_pct(pct), fmt_pct(ath_pct))
        time.sleep(0.4) 
    return grid


# --------------------------------------------------------------------------- #
# CoinGecko: treasury holdings + spot-marked P&L
# --------------------------------------------------------------------------- #

def get_treasury() -> Optional[dict]:
    def _do():
        headers = {"accept": "application/json"}
        if CG_DEMO_API_KEY:
            headers["x-cg-demo-api-key"] = CG_DEMO_API_KEY
        r = requests.get(
            COINGECKO_BASE + COINGECKO_TREASURY_PATH,
            headers=headers, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    data = _retry(_do, label="coingecko:treasury")
    if not data:
        return None

    companies = data.get("companies", []) or []
    record = next((c for c in companies if c.get("symbol") == STRATEGY_CG_SYMBOL), None)
    if record is None:  
        record = next(
            (c for c in companies
             if any(h in str(c.get("name", "")).lower() for h in STRATEGY_CG_NAME_HINTS)),
            None,
        )
    if record is None:
        log.error("Strategy not found in CoinGecko treasury list")
        return None

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


def _fmp_get(path: str, params: Optional[dict] = None):
    if not FMP_API_KEY:
        return None
    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    def _do():
        r = requests.get(f"{FMP_BASE}{path}", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code in (401, 402, 403):
            raise PermissionError(
                f"HTTP {r.status_code} on {path} "
                f"(key invalid or endpoint not on your FMP plan)"
            )
        r.raise_for_status()
        return r.json()

    return _retry(_do, label=f"fmp:{path}", attempts=2)


def fmp_sp500_symbols() -> list[str]:
    data = _fmp_get("/sp500_constituent")
    return [row["symbol"] for row in data if row.get("symbol")] if isinstance(data, list) else []


def fmp_penny_symbols() -> list[str]:
    data = _fmp_get("/stock-screener", {
        "priceLowerThan": PENNY_MAX_PRICE,
        "marketCapMoreThan": PENNY_MIN_MARKET_CAP,
        "isActivelyTrading": "true",
        "exchange": "nasdaq,nyse,amex",
        "limit": PENNY_UNIVERSE_CAP,
    })
    return [row["symbol"] for row in data if row.get("symbol")][:PENNY_UNIVERSE_CAP] if isinstance(data, list) else []


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


def rank_gainers_losers(quotes: list[dict], n: int = SCREENER_TOP_N) -> tuple[list[dict], list[dict]]:
    valid = [q for q in quotes if isinstance(q.get("change_pct"), float)]
    ranked = sorted(valid, key=lambda q: q["change_pct"], reverse=True)
    return ranked[:n], sorted(valid, key=lambda q: q["change_pct"])[:n]


def get_screener(kind: str) -> Optional[dict]:
    if not FMP_API_KEY:
        return None
    symbols = fmp_sp500_symbols() if kind == "bluechip" else fmp_penny_symbols()
    if not symbols:
        return None
    quotes = fmp_batch_quotes(symbols)
    if not quotes:
        return None
    gainers, losers = rank_gainers_losers(quotes)
    return {"gainers": gainers, "losers": losers}


# --------------------------------------------------------------------------- #
# Markdown builders
# --------------------------------------------------------------------------- #

def build_grid_table(grid: dict[str, tuple[Optional[float], Optional[float], Optional[float]]]) -> str:
    lines = ["| Ticker | Close | 24h Change | % from ATH |", "|:------:|------:|:----------:|:----------:|"]
    for tkr in MSTR_GRID_TICKERS:
        close, pct, ath_pct = grid.get(tkr, (None, None, None))
        
        arrow = ""
        if isinstance(pct, float):
            arrow = " 🔺" if pct > 0 else (" 🔻" if pct < 0 else " ▪")
            
        ath_str = fmt_pct(ath_pct)
        if isinstance(ath_pct, float) and ath_pct >= 0:
            ath_str = "ATH 🚀"
            
        lines.append(f"| **{tkr}** | {fmt_money(close)} | {fmt_pct(pct)}{arrow} | {ath_str} |")
    return "\n".join(lines)


def build_pnl_sentence(treasury: Optional[dict]) -> str:
    if not treasury or not isinstance(treasury.get("holdings"), float):
        return "_Treasury P&L unavailable — CoinGecko data could not be retrieved._"
    h = treasury["holdings"]
    entry = treasury.get("entry_value")
    current = treasury.get("current_value")
    pnl = treasury.get("pnl")
    pnl_pct = treasury.get("pnl_pct")
    if not isinstance(pnl, float):
        return (f"**Strategy holds {h:,.0f} BTC, currently valued at "
                f"{fmt_compact(current)} (cost basis unavailable).**")
    verb = "an unrealized gain" if pnl >= 0 else "an unrealized loss"
    return (
        f"**Strategy holds {h:,.0f} BTC bought for {fmt_compact(entry)}; at "
        f"CoinGecko's current valuation of {fmt_compact(current)} that is "
        f"{verb} of {fmt_compact(abs(pnl))} ({fmt_pct(pnl_pct)}).**"
    )


def _screener_table(rows: list[dict]) -> str:
    head = ["| # | Symbol | Price | Change | Mkt Cap | Company |",
            "|--:|:------:|------:|:------:|--------:|:--------|"]
    if not rows:
        return "_No qualifying stocks today._"
    for i, r in enumerate(rows, 1):
        head.append(
            f"| {i} | {r.get('symbol','?')} | {fmt_money(r.get('price'))} | "
            f"{fmt_pct(r.get('change_pct'))} | {fmt_compact(r.get('market_cap'))} | "
            f"{(r.get('name') or '')[:32]} |"
        )
    return "\n".join(head)


def build_screener_section(title: str, screener: Optional[dict], note: str = "") -> str:
    if screener is None:
        body = "_Screener unavailable._"
        return f"## {title}\n\n{body}\n"
    parts = [f"## {title}"]
    if note:
        parts.append(f"_{note}_\n")
    parts.append(f"### Top {SCREENER_TOP_N} Gainers\n\n{_screener_table(screener['gainers'])}\n")
    parts.append(f"### Top {SCREENER_TOP_N} Losers\n\n{_screener_table(screener['losers'])}\n")
    return "\n".join(parts)


def build_report(btc_price, grid, treasury, bluechip, penny) -> str:
    return "\n".join([
        f"---\ntitle: \"Saylor Infinite Money Glitch Tracker — {DATE_STR}\"\n---\n",
        f"# Saylor Infinite Money Glitch Tracker — {DATE_STR}",
        f"_Generated {TIMESTAMP_STR}_\n",
        f"**CME Bitcoin Futures (BTC=F):** {fmt_money(btc_price)}\n",
        "## The Saylor Scam Grid\n",
        build_grid_table(grid) + "\n",
        build_pnl_sentence(treasury) + "\n",
        "_P&L is marked against CoinGecko's spot valuation of the holdings._\n",
        build_screener_section("Blue Chips (S&P 500)", bluechip),
        build_screener_section("Penny Stocks", penny),
        "\n---\n_Automated report. Not investment advice._",
    ])


def list_recent_reports() -> list[Path]:
    files = sorted(REPORTS_DIR.glob("20*-*-*.md"), reverse=True)
    return files[:ROLLING_DAYS]


def build_index(grid, treasury) -> str:
    recent = list_recent_reports()
    archive_lines = [
        f"- [{p.stem}](reports/{p.stem}.html)" for p in recent
    ] or ["_No reports yet._"]
    today_link = f"reports/{DATE_STR}.html"
    
    pnl_sentence = build_pnl_sentence(treasury)
    
    # HTML formatting to force CSS injection on GitHub Pages
    if treasury and isinstance(treasury.get("pnl"), float) and treasury["pnl"] < 0:
        pnl_cleaned = pnl_sentence.strip("*")
        if "unrealized loss of " in pnl_cleaned:
            parts = pnl_cleaned.split("unrealized loss of ")
            pnl_html = f'<div class="loss-box"><strong>{parts[0]}unrealized loss of <span style="color: #ff4444; font-weight: 900; font-size: 1.1em;">{parts[1]}</span></strong></div>'
        else:
            pnl_html = f'<div class="loss-box"><strong>{pnl_cleaned}</strong></div>'
    else:
        pnl_html = f'<div class="loss-box">{pnl_sentence}</div>'

    # The !important flag forces Pages to override the standard white Jekyll theme
    style_block = """
<style>
  html, body, .markdown-body {
    background-color: #0b0b0b !important;
    color: #ffffff !important;
    font-family: "Courier New", Courier, monospace !important;
  }
  h1, h2, h3, h4 {
    color: #F7931A !important;
    text-shadow: 0 0 12px rgba(247, 147, 26, 0.4) !important;
    border-bottom: none !important;
  }
  a {
    color: #F7931A !important;
    text-decoration: none !important;
  }
  a:hover {
    text-shadow: 0 0 8px rgba(247, 147, 26, 0.8) !important;
  }
  table {
    background-color: #111 !important;
    border: 1px solid #333 !important;
    box-shadow: 0 0 15px rgba(247, 147, 26, 0.1) !important;
  }
  th {
    background-color: #1a1a1a !important;
    color: #F7931A !important;
    border-bottom: 2px solid #F7931A !important;
  }
  td {
    color: #ffffff !important;
    border-bottom: 1px solid #222 !important;
  }
  .loss-box {
    background-color: rgba(255, 68, 68, 0.05) !important;
    border-left: 4px solid #ff4444 !important;
    padding: 15px !important;
    margin: 20px 0 !important;
    color: #fff !important;
  }
  .glitch {
    color: #F7931A !important;
    font-weight: 900;
  }
</style>
"""

    return "\n".join([
        f"---\ntitle: \"Saylor Infinite Money Glitch Tracker\"\nlayout: null\n---\n",
        style_block,
        # HTML titles with orange "affect" letters injected
        f"<h1><span class='glitch'>S</span>ayl<span class='glitch'>o</span>r Inf<span class='glitch'>i</span>n<span class='glitch'>i</span>t<span class='glitch'>e</span> M<span class='glitch'>o</span>n<span class='glitch'>e</span>y Gl<span class='glitch'>i</span>tch Tr<span class='glitch'>a</span>ck<span class='glitch'>e</span>r</h1>",
        f"<p><em>Last updated {TIMESTAMP_STR}</em></p>",
        f"<h3>👉 <a href=\"{today_link}\">Today's full report — {DATE_STR}</a></h3>",
        f"<h2>The S<span class='glitch'>a</span>yl<span class='glitch'>o</span>r Sc<span class='glitch'>a</span>m G<span class='glitch'>r</span>id</h2>",
        build_grid_table(grid) + "\n",
        pnl_html + "\n",
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

    btc_price = get_btc_futures()
    grid = get_mstr_grid()
    treasury = get_treasury()
    bluechip = get_screener("bluechip")
    penny = get_screener("penny")

    report_md = build_report(btc_price, grid, treasury, bluechip, penny)
    report_path = REPORTS_DIR / f"{DATE_STR}.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("Wrote %s", report_path)

    index_md = build_index(grid, treasury)
    INDEX_FILE.write_text(index_md, encoding="utf-8")
    log.info("Wrote %s", INDEX_FILE)

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
