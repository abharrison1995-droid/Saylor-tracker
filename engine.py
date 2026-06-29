#!/usr/bin/env python3
"""
engine.py — Headless Strategy (MicroStrategy) + market tracker.

Fetches market/crypto data and regenerates a daily Markdown report plus a
rolling front page. Designed to run unattended in GitHub Actions and never
crash the workflow: every data source is wrapped so that a single failure
degrades one section (shows "N/A") instead of aborting the whole run.

------------------------------------------------------------------------------
ENGINEERING NOTES — where this deviates from the original spec, and why
------------------------------------------------------------------------------
1. Penny-stock screener:
   yfinance is a *quote fetcher*, not a *screener* — it cannot enumerate "all
   stocks under $5 with > $100M market cap". That filter is delegated to the
   Financial Modeling Prep (FMP) /stock-screener endpoint, which supports
   priceLowerThan + marketCapMoreThan natively. Same applies to ranking the
   S&P 500 by daily move. Requires a free FMP_API_KEY (degrades gracefully if
   absent). yfinance is kept only for the ~6 hand-picked tickers, where it's
   reliable even from Actions' datacenter IPs.

2. Treasury P&L:
   The spec marked a *spot* BTC holding against the *futures* price (BTC=F).
   Those differ (basis/contango), so that's a methodology bug. CoinGecko's
   treasury endpoint already returns `total_current_value_usd` — its own
   mark-to-spot valuation — so P&L is simply current_value - entry_value.
   BTC=F is still shown, but purely as a headline number.

3. CoinGecko changed the record:
   The company is now listed as name "Strategy", symbol "MSTR.US" (post
   rebrand). We match on symbol first, then fall back to a name search, so a
   future rename doesn't silently zero out the P&L.
------------------------------------------------------------------------------
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
except Exception as exc:  # pragma: no cover - import guard for clearer logs
    raise SystemExit(f"yfinance failed to import — is it installed? ({exc})")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BTC_FUTURES_TICKER = "BTC=F"            # CME Bitcoin front-month future (headline only)
MSTR_GRID_TICKERS = ["MSTR", "STRC", "STRK", "STRF", "STRD"]

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_TREASURY_PATH = "/companies/public_treasury/bitcoin"
STRATEGY_CG_SYMBOL = "MSTR.US"          # CoinGecko's symbol for Strategy/MicroStrategy
STRATEGY_CG_NAME_HINTS = ("strategy", "microstrategy")

# FMP. The free tier uses the /api/v3 paths below. If you upgrade to a newer
# key that only serves the "stable" surface, flip FMP_BASE to
# "https://financialmodelingprep.com/stable" and adjust the paths.
FMP_BASE = "https://financialmodelingprep.com/api/v3"
PENNY_MAX_PRICE = 5.0
PENNY_MIN_MARKET_CAP = 100_000_000      # > $100M
PENNY_UNIVERSE_CAP = 1000               # cap symbols sent to the quote endpoint
SCREENER_TOP_N = 20                     # top/bottom N gainers/losers
QUOTE_CHUNK = 50                        # symbols per batch /quote call

# Secrets (read from environment; set as GitHub Actions secrets)
FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()
CG_DEMO_API_KEY = os.environ.get("CG_DEMO_API_KEY", "").strip()  # optional

REPORTS_DIR = Path("reports")
INDEX_FILE = Path("index.md")
ROLLING_DAYS = 30
HTTP_TIMEOUT = 20
RETRY_ATTEMPTS = 3

NOW_UTC = datetime.now(timezone.utc)
DATE_STR = NOW_UTC.date().isoformat()   # YYYY-MM-DD
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
    """Run fn() with linear backoff. Returns fn()'s value or None on failure."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to swallow & retry
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
    """Human-readable large dollar amounts: $75.02B, $123.4M."""
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
    """Parse numbers that may arrive as floats or messy strings like '(+2.3%)'."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            return float(m.group())
    return None


# --------------------------------------------------------------------------- #
# yfinance: BTC=F headline + the Strategy grid
# --------------------------------------------------------------------------- #

def fetch_quote_yf(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """Return (last_close, pct_change_vs_prev_close) for a ticker, or (None, None)."""
    def _do():
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            raise ValueError("empty history")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise ValueError("no closes")
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
        pct = ((last / prev) - 1.0) * 100.0 if prev else None
        return last, pct

    result = _retry(_do, label=f"yfinance:{ticker}")
    return result if result is not None else (None, None)


def get_btc_futures() -> Optional[float]:
    price, _ = fetch_quote_yf(BTC_FUTURES_TICKER)
    log.info("BTC=F: %s", fmt_money(price))
    return price


def get_mstr_grid() -> dict[str, tuple[Optional[float], Optional[float]]]:
    grid: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for tkr in MSTR_GRID_TICKERS:
        close, pct = fetch_quote_yf(tkr)
        grid[tkr] = (close, pct)
        log.info("%-5s close=%s change=%s", tkr, fmt_money(close), fmt_pct(pct))
        time.sleep(0.4)  # be gentle with the source
    return grid


# --------------------------------------------------------------------------- #
# CoinGecko: treasury holdings + spot-marked P&L
# --------------------------------------------------------------------------- #

def get_treasury() -> Optional[dict]:
    """
    Fetch Strategy's BTC treasury figures from CoinGecko.
    Returns {holdings, entry_value, current_value, pnl, pnl_pct} or None.
    """
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
    if record is None:  # fallback: match by name if the symbol ever changes
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

    log.info("Treasury: %s BTC | entry %s | current %s",
             f"{holdings:,.0f}" if holdings else "N/A",
             fmt_compact(entry), fmt_compact(current))
    return {
        "holdings": holdings,
        "entry_value": entry,
        "current_value": current,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }


# --------------------------------------------------------------------------- #
# FMP: screeners (blue chips via S&P 500 constituents, penny stocks via filter)
# --------------------------------------------------------------------------- #

def _fmp_get(path: str, params: Optional[dict] = None):
    """GET an FMP endpoint. Returns parsed JSON, or None (logs the reason)."""
    if not FMP_API_KEY:
        return None
    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    def _do():
        r = requests.get(f"{FMP_BASE}{path}", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code in (401, 402, 403):
            # Auth/plan problem — don't retry, surface a clear message.
            raise PermissionError(
                f"HTTP {r.status_code} on {path} "
                f"(key invalid or endpoint not on your FMP plan)"
            )
        r.raise_for_status()
        return r.json()

    return _retry(_do, label=f"fmp:{path}", attempts=2)


def fmp_sp500_symbols() -> list[str]:
    data = _fmp_get("/sp500_constituent")
    if not isinstance(data, list):
        return []
    return [row["symbol"] for row in data if row.get("symbol")]


def fmp_penny_symbols() -> list[str]:
    data = _fmp_get("/stock-screener", {
        "priceLowerThan": PENNY_MAX_PRICE,
        "marketCapMoreThan": PENNY_MIN_MARKET_CAP,
        "isActivelyTrading": "true",
        "exchange": "nasdaq,nyse,amex",
        "limit": PENNY_UNIVERSE_CAP,
    })
    if not isinstance(data, list):
        return []
    return [row["symbol"] for row in data if row.get("symbol")][:PENNY_UNIVERSE_CAP]


def fmp_batch_quotes(symbols: list[str]) -> list[dict]:
    """Fetch quotes for many symbols in chunks. Returns normalized dicts."""
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


def rank_gainers_losers(quotes: list[dict], n: int = SCREENER_TOP_N
                        ) -> tuple[list[dict], list[dict]]:
    valid = [q for q in quotes if isinstance(q.get("change_pct"), float)]
    ranked = sorted(valid, key=lambda q: q["change_pct"], reverse=True)
    gainers = ranked[:n]
    losers = sorted(valid, key=lambda q: q["change_pct"])[:n]
    return gainers, losers


def get_screener(kind: str) -> Optional[dict]:
    """kind in {'bluechip', 'penny'}. Returns {gainers, losers} or None."""
    if not FMP_API_KEY:
        log.warning("Skipping %s screener — FMP_API_KEY not set", kind)
        return None
    symbols = fmp_sp500_symbols() if kind == "bluechip" else fmp_penny_symbols()
    if not symbols:
        log.error("%s screener: no symbols returned (check FMP plan/limits)", kind)
        return None
    log.info("%s screener: %d symbols -> fetching quotes", kind, len(symbols))
    quotes = fmp_batch_quotes(symbols)
    if not quotes:
        log.error("%s screener: no quotes returned", kind)
        return None
    gainers, losers = rank_gainers_losers(quotes)
    return {"gainers": gainers, "losers": losers}


# --------------------------------------------------------------------------- #
# Markdown builders
# --------------------------------------------------------------------------- #

def build_grid_table(grid: dict[str, tuple[Optional[float], Optional[float]]]) -> str:
    lines = ["| Ticker | Close | 24h Change |", "|:------:|------:|:----------:|"]
    for tkr in MSTR_GRID_TICKERS:
        close, pct = grid.get(tkr, (None, None))
        arrow = ""
        if isinstance(pct, float):
            arrow = " 🔺" if pct > 0 else (" 🔻" if pct < 0 else " ▪")
        lines.append(f"| **{tkr}** | {fmt_money(close)} | {fmt_pct(pct)}{arrow} |")
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
        body = ("_Screener unavailable. Set the `FMP_API_KEY` secret; if it is "
                "set, this endpoint may not be included in your FMP plan._")
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
        "_P&L is marked against CoinGecko's spot valuation of the holdings "
        "(`total_current_value_usd`); BTC=F above is the front-month future, "
        "shown as a headline only._\n",
        build_screener_section(
            "Blue Chips (S&P 500)", bluechip,
            note="Top movers among current S&P 500 constituents."),
        build_screener_section(
            "Penny Stocks", penny,
            note=f"Priced under ${PENNY_MAX_PRICE:.0f} with market cap over "
                 f"{fmt_compact(PENNY_MIN_MARKET_CAP)}."),
        "\n---\n_Automated report. Not investment advice._",
    ])


def list_recent_reports() -> list[Path]:
    """Most-recent-first list of report files (ISO names sort lexically)."""
    files = sorted(REPORTS_DIR.glob("20*-*-*.md"), reverse=True)
    return files[:ROLLING_DAYS]


def build_index(grid, treasury) -> str:
    recent = list_recent_reports()
    archive_lines = [
        f"- [{p.stem}](reports/{p.stem}.html)" for p in recent
    ] or ["_No reports yet._"]
    today_link = f"reports/{DATE_STR}.html"
    
    pnl_sentence = build_pnl_sentence(treasury)
    
    # Safely convert raw markdown text inside the P&L block to structured HTML
    # for cleaner rendering inside our CSS loss container.
    if treasury and isinstance(treasury.get("pnl"), float) and treasury["pnl"] < 0:
        pnl_cleaned = pnl_sentence.strip("*")
        if "unrealized loss of " in pnl_cleaned:
            parts = pnl_cleaned.split("unrealized loss of ")
            pnl_html = f'<div class="loss-box"><strong>{parts[0]}unrealized loss of <span style="color: #ff4444;">{parts[1]}</span></strong></div>'
        else:
            pnl_html = f'<div class="loss-box"><strong>{pnl_cleaned}</strong></div>'
    else:
        pnl_html = f'<div class="loss-box">{pnl_sentence}</div>'

    style_block = """<style>
  body {
    background-color: #0b0b0b;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  h1, h2, h3 {
    color: #F7931A;
    text-shadow: 0 0 10px rgba(247, 147, 26, 0.3);
  }
  a {
    color: #F7931A;
    text-decoration: none;
    transition: all 0.2s ease-in-out;
  }
  a:hover {
    text-shadow: 0 0 8px rgba(247, 147, 26, 0.8);
  }
  table {
    width: 100%;
    max-width: 600px;
    background-color: #141414;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    border-collapse: collapse;
  }
  th {
    color: #F7931A;
    background-color: #1f1f1f;
    border-bottom: 2px solid #333;
    padding: 10px;
  }
  td {
    border-bottom: 1px solid #222;
    padding: 10px;
  }
  .loss-box {
    background-color: #1a1a1a;
    border-left: 4px solid #ff4444;
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
  }
  hr {
    border: 0;
    border-top: 1px solid #333;
    margin: 30px 0 15px 0;
  }
</style>"""

    return "\n".join([
        f"---\ntitle: \"Saylor Infinite Money Glitch Tracker\"\n---\n",
        style_block,
        f"# 🚀 Saylor Infinite Money Glitch Tracker",
        f"_Last updated {TIMESTAMP_STR}_\n",
        f"### 👉 [Today's full report — {DATE_STR}]({today_link})\n",
        "## The Saylor Scam Grid\n",
        build_grid_table(grid) + "\n",
        pnl_html + "\n",
        f"## Archive — last {ROLLING_DAYS} days\n",
        "\n".join(archive_lines) + "\n",
        "---\n<p style=\"color: #666; font-size: 0.85em; text-align: center;\"><em>Automated. Not investment advice.</em></p>",
    ])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info("=== Saylor Tracker run: %s ===", DATE_STR)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Each call is internally guarded and returns None/partial on failure.
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
    log.info("Wrote %s (archive: %d reports)", INDEX_FILE, len(list_recent_reports()))

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
