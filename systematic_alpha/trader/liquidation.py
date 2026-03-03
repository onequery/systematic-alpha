from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import TraderConfig
from systematic_alpha.trader.execution import place_market_order
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.sync import AccountSyncService
from systematic_alpha.trader.telegram import TelegramClient


KST = ZoneInfo("Asia/Seoul")


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


@dataclass
class LiquidationOutcome:
    market: str
    trade_date: str
    attempted: int
    sent: int
    rejected: int
    remaining: int
    reject_reasons: List[str]


def run_liquidation(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    sync_service: AccountSyncService,
    telegram: TelegramClient,
    market: str,
    trade_date: str,
    phase: str,
) -> LiquidationOutcome:
    mk = str(market).upper()
    pre = sync_service.sync_account(market_scope=mk, strict=cfg.strict_sync, trade_date=trade_date)
    positions = [
        row
        for row in pre.positions
        if str(row.get("market", "")).upper() == mk and float(row.get("quantity", 0.0) or 0.0) > 0
    ]

    sent = 0
    rejected = 0
    reject_reasons: List[str] = []
    for pos in positions:
        symbol = str(pos.get("symbol", "")).upper()
        qty = int(float(pos.get("quantity", 0.0) or 0.0))
        if qty <= 0:
            continue
        ok, reason, broker_order_id, payload = place_market_order(
            cfg=cfg,
            market=mk,
            side="SELL",
            symbol=symbol,
            quantity=qty,
        )
        status = "SENT" if ok else "REJECTED"
        storage.insert_order(
            trade_date=trade_date,
            market=mk,
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            quantity=qty,
            reference_price=float(pos.get("avg_price", 0.0) or 0.0),
            status=status,
            reject_reason="" if ok else reason,
            broker_order_id=broker_order_id,
            broker_response=payload if isinstance(payload, dict) else {},
            submitted_at=_now_iso(),
        )
        if ok:
            sent += 1
        else:
            rejected += 1
            reject_reasons.append(f"{symbol}:{reason}")

    post = sync_service.sync_account(market_scope=mk, strict=cfg.strict_sync, trade_date=trade_date)
    remaining = len(
        [
            row
            for row in post.positions
            if str(row.get("market", "")).upper() == mk and float(row.get("quantity", 0.0) or 0.0) > 0
        ]
    )
    storage.log_event(
        "liquidation_phase",
        {
            "market": mk,
            "trade_date": trade_date,
            "phase": phase,
            "attempted": len(positions),
            "sent": sent,
            "rejected": rejected,
            "remaining": remaining,
            "reject_reasons": reject_reasons[:5],
        },
    )

    if len(positions) > 0:
        msg = (
            f"[이벤트] [Trader] 청산 실행\n시장={mk}\n일자={trade_date}\n"
            f"단계={phase}\n시도={len(positions)}\n접수={sent}\n거부={rejected}\n잔여={remaining}"
        )
        if reject_reasons:
            msg += "\n거부사유(최대3건):\n- " + "\n- ".join(reject_reasons[:3])
        telegram.send(msg)
    if phase == "final_check" and remaining > 0:
        telegram.send(
            f"[Action required] [Trader] 청산 실패\n시장={mk}\n일자={trade_date}\n잔여포지션={remaining}"
        )

    return LiquidationOutcome(
        market=mk,
        trade_date=trade_date,
        attempted=len(positions),
        sent=sent,
        rejected=rejected,
        remaining=remaining,
        reject_reasons=reject_reasons,
    )

