from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from systematic_alpha.credentials import load_credentials
from systematic_alpha.helpers import pick_first
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.trader.config import TraderConfig
from systematic_alpha.trader.realtime import SignalIntent
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.sync import AccountSyncService, summarize_positions
from systematic_alpha.trader.telegram import TelegramClient


KST = ZoneInfo("Asia/Seoul")


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _api_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "non_dict_response"
    rt_cd = str(payload.get("rt_cd", "") or "").strip()
    msg_cd = str(payload.get("msg_cd", "") or "").strip()
    msg1 = str(payload.get("msg1", "") or "").strip()
    if rt_cd and rt_cd != "0":
        out = [f"rt_cd={rt_cd}"]
        if msg_cd:
            out.append(f"msg_cd={msg_cd}")
        if msg1:
            out.append(f"msg1={msg1}")
        return ", ".join(out)
    return ""


def _extract_broker_order_id(payload: Dict[str, Any]) -> str:
    output = payload.get("output", {})
    if isinstance(output, dict):
        odno = pick_first(output, ("ODNO", "odno", "ord_no", "order_no"))
        if odno not in (None, ""):
            return str(odno)
    odno = pick_first(payload, ("ODNO", "odno", "ord_no", "order_no"))
    if odno not in (None, ""):
        return str(odno)
    return ""


def _new_broker(cfg: TraderConfig, market: str, exchange: str = ""):
    key, secret, acc_no, _ = load_credentials(None)
    mojito = import_mojito_module()
    if str(market).upper() == "KR":
        return mojito.KoreaInvestment(api_key=key, api_secret=secret, acc_no=acc_no, mock=cfg.use_mock)

    ex = str(exchange or "").upper()
    ex_label = {"NASD": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스"}.get(ex, "뉴욕")
    return mojito.KoreaInvestment(
        api_key=key,
        api_secret=secret,
        acc_no=acc_no,
        exchange=ex_label,
        mock=cfg.use_mock,
    )


def place_market_order(
    *,
    cfg: TraderConfig,
    market: str,
    side: str,
    symbol: str,
    quantity: int,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    mk = str(market).upper()
    sd = str(side).upper()
    exchanges = ["KR"] if mk == "KR" else list(cfg.us_sync_exchanges or ["NYSE", "AMEX"])
    last_error = "broker_unavailable"
    last_payload: Dict[str, Any] = {}

    for ex in exchanges:
        try:
            broker = _new_broker(cfg, mk, ex)
        except Exception as exc:
            last_error = f"broker_init_failed:{ex}:{repr(exc)}"
            continue
        try:
            if sd == "BUY":
                payload = broker.create_market_buy_order(symbol, int(quantity))
            else:
                payload = broker.create_market_sell_order(symbol, int(quantity))
        except Exception as exc:
            last_error = f"order_exception:{ex}:{repr(exc)}"
            continue
        if not isinstance(payload, dict):
            last_error = f"invalid_response:{ex}"
            last_payload = {}
            continue
        last_payload = payload
        err = _api_error_text(payload)
        if err:
            last_error = f"{ex}:{err}"
            continue
        return True, "", _extract_broker_order_id(payload), payload

    return False, last_error, "", last_payload


@dataclass
class ExecutionOutcome:
    market: str
    trade_date: str
    proposed: int
    sent: int
    rejected: int
    skipped: int
    reject_reasons: List[str]
    order_ids: List[int]


def _fx_for_market(positions: List[Dict[str, Any]], market: str) -> float:
    mk = str(market).upper()
    if mk == "KR":
        return 1.0
    fx_candidates = [
        float(row.get("fx_rate", 0.0) or 0.0)
        for row in positions
        if str(row.get("market", "")).upper() == "US" and float(row.get("fx_rate", 0.0) or 0.0) > 0
    ]
    if fx_candidates:
        return max(fx_candidates)
    return 1300.0


def execute_entry_intents(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    sync_service: AccountSyncService,
    telegram: TelegramClient,
    market: str,
    trade_date: str,
    intents: List[SignalIntent],
) -> ExecutionOutcome:
    mk = str(market).upper()
    pre = sync_service.sync_account(market_scope=mk, strict=cfg.strict_sync, trade_date=trade_date)
    if not pre.ok:
        storage.log_event(
            "execution_skipped_sync_failed",
            {"market": mk, "trade_date": trade_date, "errors": pre.errors, "reason": pre.reason},
        )
        return ExecutionOutcome(mk, trade_date, len(intents), 0, 0, len(intents), list(pre.errors), [])

    budget = storage.get_day_budget(trade_date)
    if not budget:
        storage.log_event("execution_skipped_no_budget", {"market": mk, "trade_date": trade_date})
        return ExecutionOutcome(mk, trade_date, len(intents), 0, 0, len(intents), ["budget_missing"], [])

    if mk == "KR":
        max_market = int(cfg.max_positions_kr)
    else:
        max_market = int(cfg.max_positions_us)
    max_total = int(cfg.max_positions_total)

    positions = pre.positions
    holdings_by_market = {
        "KR": summarize_positions(positions, "KR"),
        "US": summarize_positions(positions, "US"),
    }
    market_count = len([k for k, v in holdings_by_market.get(mk, {}).items() if float(v) > 0])
    total_count = len(
        [
            (m, s)
            for m in ("KR", "US")
            for s, q in holdings_by_market.get(m, {}).items()
            if float(q) > 0
        ]
    )
    available_cash = float(pre.cash_krw or 0.0)
    budget_per_trade = float(budget.get("budget_per_trade", 0.0) or 0.0)
    fx_rate = _fx_for_market(positions, mk)

    sent = 0
    rejected = 0
    skipped = 0
    reject_reasons: List[str] = []
    order_ids: List[int] = []

    for intent in intents:
        if market_count >= max_market or total_count >= max_total:
            skipped += 1
            continue
        if intent.symbol in holdings_by_market.get(mk, {}):
            skipped += 1
            continue

        local_price = float(intent.last_price)
        if mk == "US":
            est_cost_krw = local_price * fx_rate
        else:
            est_cost_krw = local_price
        qty = int(math.floor(budget_per_trade / est_cost_krw)) if est_cost_krw > 0 else 0
        if qty <= 0:
            skipped += 1
            reject_reasons.append(f"{intent.symbol}:quantity_zero")
            continue

        expected_order_cost = est_cost_krw * qty
        if available_cash < expected_order_cost:
            rejected += 1
            reason = "INSUFFICIENT_CASH"
            reject_reasons.append(f"{intent.symbol}:{reason}")
            order_id = storage.insert_order(
                trade_date=trade_date,
                market=mk,
                symbol=intent.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=qty,
                reference_price=float(intent.last_price),
                status="REJECTED",
                reject_reason=reason,
                broker_order_id="",
                broker_response={"reason": reason, "available_cash_krw": available_cash},
                submitted_at=_now_iso(),
            )
            order_ids.append(order_id)
            telegram.send(
                f"[Action required] [Trader] 주문 거부\n시장={mk}\n종목={intent.symbol}\n사유=INSUFFICIENT_CASH"
            )
            continue

        ok, reason, broker_order_id, payload = place_market_order(
            cfg=cfg,
            market=mk,
            side="BUY",
            symbol=intent.symbol,
            quantity=qty,
        )
        status = "SENT" if ok else "REJECTED"
        reject_reason = "" if ok else reason
        order_id = storage.insert_order(
            trade_date=trade_date,
            market=mk,
            symbol=intent.symbol,
            side="BUY",
            order_type="MARKET",
            quantity=qty,
            reference_price=float(intent.last_price),
            status=status,
            reject_reason=reject_reason,
            broker_order_id=broker_order_id,
            broker_response=payload if isinstance(payload, dict) else {},
            submitted_at=_now_iso(),
        )
        order_ids.append(order_id)

        if ok:
            sent += 1
            available_cash = max(0.0, available_cash - expected_order_cost)
            market_count += 1
            total_count += 1
            holdings_by_market.setdefault(mk, {})[intent.symbol] = float(qty)
            storage.mark_entered_today(trade_date=trade_date, market=mk, symbol=intent.symbol, entered=True)
        else:
            rejected += 1
            reject_reasons.append(f"{intent.symbol}:{reason}")

    post = sync_service.sync_account(market_scope=mk, strict=cfg.strict_sync, trade_date=trade_date)
    storage.log_event(
        "orders_executed",
        {
            "market": mk,
            "trade_date": trade_date,
            "sent": sent,
            "rejected": rejected,
            "skipped": skipped,
            "sync_post_ok": post.ok,
            "sync_post_errors": post.errors,
            "order_ids": order_ids,
        },
    )
    if sent > 0 or rejected > 0:
        lines = []
        if sent:
            lines.append(f"접수={sent}")
        if rejected:
            lines.append(f"거부={rejected}")
        msg = f"[보고] [Trader] 거래 실행\n시장={mk}\n일자={trade_date}\n" + ", ".join(lines)
        if reject_reasons:
            msg += "\n거부사유(최대3건):\n- " + "\n- ".join(reject_reasons[:3])
        telegram.send(msg)

    return ExecutionOutcome(
        market=mk,
        trade_date=trade_date,
        proposed=len(intents),
        sent=sent,
        rejected=rejected,
        skipped=skipped,
        reject_reasons=reject_reasons,
        order_ids=order_ids,
    )

