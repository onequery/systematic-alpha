from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import TraderConfig
from systematic_alpha.trader.selector_bridge import make_selector
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.sync import summarize_positions


KST = ZoneInfo("Asia/Seoul")


@dataclass
class SignalIntent:
    market: str
    symbol: str
    name: str
    last_price: float
    breakout_price: float
    today_open: float
    candidate_rank: int


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _watch_symbols(storage: TraderStorage, market: str, trade_date: str) -> Dict[str, Set[str]]:
    candidates = storage.list_candidate_symbols(trade_date, market)
    candidate_symbols = {str(row.get("symbol", "")).upper() for row in candidates if str(row.get("symbol", "")).strip()}

    latest_scope = "ALL"
    if market in {"KR", "US"}:
        latest_scope = market
    positions = storage.latest_positions(market_scope=latest_scope, market=market)
    holding_symbols = {
        str(row.get("symbol", "")).upper()
        for row in positions
        if float(row.get("quantity", 0.0) or 0.0) > 0
    }
    return {
        "candidates": candidate_symbols,
        "holdings": holding_symbols,
        "watch": set(candidate_symbols) | set(holding_symbols),
    }


def refresh_market_prices(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    market: str,
    trade_date: str,
) -> Dict[str, Any]:
    mk = str(market).upper()
    market_slug = str(mk).lower()
    session_root = cfg.out_dir / market_slug / trade_date
    output_json_path = session_root / "results" / f"{market_slug}_refresh_{trade_date}.json"
    analytics_dir = session_root / "analytics"
    symbol_sets = _watch_symbols(storage, mk, trade_date)
    watch_list = sorted(symbol_sets["watch"])
    try:
        selector = make_selector(
            cfg=cfg,
            market=mk,
            output_json_path=str(output_json_path),
            analytics_dir=str(analytics_dir),
        )
    except Exception as exc:
        return {
            "market": mk,
            "trade_date": trade_date,
            "watch_count": len(watch_list),
            "candidate_count": len(symbol_sets["candidates"]),
            "holding_count": len(symbol_sets["holdings"]),
            "updated": 0,
            "failed": len(watch_list),
            "error": repr(exc),
            "ts": _now_iso(),
        }
    updated = 0
    failed = 0

    for symbol in watch_list:
        snap = None
        try:
            snap = selector.fetch_price_snapshot(symbol, use_cache=False)
        except Exception:
            snap = None

        if not snap:
            failed += 1
            continue
        last_price = snap.get("price")
        today_open = snap.get("open")
        try:
            storage.update_symbol_snapshot(
                trade_date=trade_date,
                market=mk,
                symbol=symbol,
                last_price=float(last_price) if last_price is not None else None,
                today_open=float(today_open) if today_open is not None else None,
                breakout_k=cfg.k,
            )
            updated += 1
        except Exception:
            failed += 1

    return {
        "market": mk,
        "trade_date": trade_date,
        "watch_count": len(watch_list),
        "candidate_count": len(symbol_sets["candidates"]),
        "holding_count": len(symbol_sets["holdings"]),
        "updated": updated,
        "failed": failed,
        "ts": _now_iso(),
    }


def collect_breakout_intents(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    market: str,
    trade_date: str,
) -> List[SignalIntent]:
    mk = str(market).upper()
    rows = storage.list_candidate_symbols(trade_date, mk)
    intents: List[SignalIntent] = []
    for row in rows:
        entered = bool(row.get("entered_today", False))
        if entered:
            continue
        last_price = row.get("last_price")
        breakout = row.get("breakout_price")
        today_open = row.get("today_open")
        try:
            last_val = float(last_price)
            breakout_val = float(breakout)
            open_val = float(today_open)
        except Exception:
            continue
        if last_val < breakout_val:
            continue
        intents.append(
            SignalIntent(
                market=mk,
                symbol=str(row.get("symbol", "")).upper(),
                name=str(row.get("name", "") or ""),
                last_price=last_val,
                breakout_price=breakout_val,
                today_open=open_val,
                candidate_rank=int(row.get("candidate_rank", 0) or 0),
            )
        )
    intents.sort(key=lambda x: (x.candidate_rank, x.symbol))
    return intents
