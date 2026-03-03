from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import TraderConfig
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.telegram import TelegramClient


KST = ZoneInfo("Asia/Seoul")


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_daily_report(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    telegram: TelegramClient,
    trade_date: str,
) -> Dict[str, Any]:
    orders = storage.list_orders(trade_date)
    fills = storage.list_fills(trade_date)
    last_snapshot = storage.latest_account_snapshot("ALL")
    positions_left = storage.latest_positions(market_scope="ALL")
    position_count = len([p for p in positions_left if float(p.get("quantity", 0.0) or 0.0) > 0])

    buy_notional = 0.0
    sell_notional = 0.0
    for fill in fills:
        qty = float(fill.get("fill_quantity", 0.0) or 0.0)
        price = float(fill.get("fill_price", 0.0) or 0.0)
        fx = float(fill.get("fx_rate", 1.0) or 1.0)
        side = str(fill.get("side", "")).upper()
        val = qty * price * fx
        if side == "BUY":
            buy_notional += val
        elif side == "SELL":
            sell_notional += val

    payload = {
        "trade_date": trade_date,
        "generated_at": _now_iso(),
        "orders": {
            "count": len(orders),
            "sent": len([o for o in orders if str(o.get("status", "")).upper() == "SENT"]),
            "rejected": len([o for o in orders if str(o.get("status", "")).upper() == "REJECTED"]),
            "filled": len([o for o in orders if str(o.get("status", "")).upper() == "FILLED"]),
        },
        "fills": {
            "count": len(fills),
            "buy_notional_krw": buy_notional,
            "sell_notional_krw": sell_notional,
            "cashflow_krw": sell_notional - buy_notional,
        },
        "account": {
            "cash_krw": float(last_snapshot.get("cash_krw", 0.0) or 0.0) if last_snapshot else 0.0,
            "equity_krw": float(last_snapshot.get("equity_krw", 0.0) or 0.0) if last_snapshot else 0.0,
            "position_count": position_count,
        },
    }
    storage.upsert_daily_report(trade_date, payload)

    out_dir = cfg.out_dir / trade_date
    _ensure_dir(out_dir)
    json_path = out_dir / f"session_close_report_{trade_date}.json"
    md_path = out_dir / f"session_close_report_{trade_date}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# Trader Daily Report {trade_date}",
                "",
                f"- generated_at: {payload['generated_at']}",
                f"- orders: {payload['orders']}",
                f"- fills: {payload['fills']}",
                f"- account: {payload['account']}",
            ]
        ),
        encoding="utf-8",
    )
    storage.log_event("daily_report_generated", {"trade_date": trade_date, "json_path": str(json_path)})

    telegram.send(
        "[보고] [Trader] 일일 결과 요약\n"
        f"일자={trade_date}\n"
        f"주문={payload['orders']['count']} (거부={payload['orders']['rejected']})\n"
        f"체결={payload['fills']['count']}\n"
        f"평가자산={payload['account']['equity_krw']:.0f}"
    )
    return payload
