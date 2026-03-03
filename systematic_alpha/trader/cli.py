from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import TraderConfig, load_trader_config
from systematic_alpha.trader.execution import execute_entry_intents
from systematic_alpha.trader.liquidation import run_liquidation
from systematic_alpha.trader.precompute import precompute_all_markets, precompute_market
from systematic_alpha.trader.realtime import collect_breakout_intents, refresh_market_prices
from systematic_alpha.trader.report import generate_daily_report
from systematic_alpha.trader.scheduler import (
    is_market_open,
    liquidation_phase,
    should_run_budget_snapshot,
    should_run_precompute,
    trade_date_et,
    trade_date_kst,
)
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.sync import AccountSyncService
from systematic_alpha.trader.telegram import TelegramClient


KST = ZoneInfo("Asia/Seoul")


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _parse_markets(value: str) -> List[str]:
    raw = str(value or "ALL").strip().upper()
    if raw in {"ALL", "*"}:
        return ["KR", "US"]
    parts = [x.strip().upper() for x in raw.split(",") if x.strip()]
    out = []
    seen = set()
    for item in parts:
        if item not in {"KR", "US"}:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _ensure_trader_paths(cfg: TraderConfig) -> Dict[str, Path]:
    paths = {
        "state": cfg.project_root / "state" / "trader",
        "out": cfg.project_root / "out" / "trader",
        "logs": cfg.project_root / "logs" / "trader",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _build_runtime(cfg: TraderConfig):
    _ensure_trader_paths(cfg)
    db_path = cfg.project_root / "state" / "trader" / "trader.sqlite"
    storage = TraderStorage(db_path)
    telegram = TelegramClient(cfg)
    sync = AccountSyncService(cfg, storage)
    return storage, telegram, sync


def _snapshot_budget(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    sync: AccountSyncService,
    trade_date: str,
) -> Dict[str, object]:
    res = sync.sync_account("ALL", strict=cfg.strict_sync, trade_date=trade_date)
    if not res.ok:
        payload = {"trade_date": trade_date, "ok": False, "reason": res.reason, "errors": res.errors}
        storage.log_event("budget_snapshot_failed", payload)
        return payload

    day_start_cash = float(res.cash_krw)
    budget_per_trade = float(day_start_cash * float(cfg.per_trade_ratio))
    payload = {
        "trade_date": trade_date,
        "captured_at": _now_iso(),
        "cash_krw": day_start_cash,
        "equity_krw": float(res.equity_krw),
        "per_trade_ratio": float(cfg.per_trade_ratio),
        "budget_per_trade": budget_per_trade,
        "snapshot_id": int(res.snapshot_id),
    }
    storage.upsert_day_budget(
        trade_date=trade_date,
        day_start_cash_snapshot_total=day_start_cash,
        per_trade_ratio=cfg.per_trade_ratio,
        budget_per_trade=budget_per_trade,
        captured_at=payload["captured_at"],
        payload=payload,
    )
    storage.log_event("budget_snapshot_captured", payload)
    return {"ok": True, **payload}


def _run_market_cycle(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    sync: AccountSyncService,
    telegram: TelegramClient,
    market: str,
    trade_date: str,
) -> Dict[str, object]:
    mk = str(market).upper()
    refresh = refresh_market_prices(cfg=cfg, storage=storage, market=mk, trade_date=trade_date)
    storage.log_event("realtime_refresh", refresh)

    market_filter = storage.get_market_filter(trade_date, mk)
    trading_enabled = bool(market_filter and market_filter.get("trading_enabled"))
    if not trading_enabled:
        payload = {
            "market": mk,
            "trade_date": trade_date,
            "status": "SKIPPED_MARKET_FILTER_OFF",
            "filter": market_filter or {},
        }
        storage.log_event("cycle_skipped", payload)
        return payload

    intents = collect_breakout_intents(cfg=cfg, storage=storage, market=mk, trade_date=trade_date)
    if not intents:
        payload = {"market": mk, "trade_date": trade_date, "status": "NO_INTENT", "intent_count": 0}
        storage.log_event("cycle_no_intent", payload)
        return payload

    outcome = execute_entry_intents(
        cfg=cfg,
        storage=storage,
        sync_service=sync,
        telegram=telegram,
        market=mk,
        trade_date=trade_date,
        intents=intents,
    )
    payload = {
        "market": mk,
        "trade_date": trade_date,
        "status": "EXECUTED",
        "intent_count": len(intents),
        "sent": outcome.sent,
        "rejected": outcome.rejected,
        "skipped": outcome.skipped,
        "reject_reasons": outcome.reject_reasons[:5],
    }
    storage.log_event("cycle_executed", payload)
    return payload


def _archive_and_reset(cfg: TraderConfig) -> Dict[str, object]:
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    archive_root = cfg.project_root / "archive" / f"cutover_trader_{stamp}"
    archive_root.mkdir(parents=True, exist_ok=True)

    move_targets = [
        cfg.project_root / "state" / "agent_lab",
        cfg.project_root / "out" / "agent_lab",
        cfg.project_root / "out" / "kr",
        cfg.project_root / "out" / "us",
        cfg.project_root / "logs" / "agent_lab",
        cfg.project_root / "logs" / "kr",
        cfg.project_root / "logs" / "us",
        cfg.project_root / "state" / "trader",
        cfg.project_root / "out" / "trader",
        cfg.project_root / "logs" / "trader",
    ]
    moved = []
    for src in move_targets:
        if not src.exists():
            continue
        dst = archive_root / src.relative_to(cfg.project_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append({"from": str(src), "to": str(dst)})

    cron_dir = cfg.project_root / "logs" / "cron"
    if cron_dir.exists():
        for file in cron_dir.glob("agent_*"):
            dst = archive_root / "logs" / "cron" / file.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(dst))
            moved.append({"from": str(file), "to": str(dst)})

    _ensure_trader_paths(cfg)
    storage = TraderStorage(cfg.project_root / "state" / "trader" / "trader.sqlite")
    storage.log_event("cutover_reset", {"archive_root": str(archive_root), "moved_count": len(moved)})
    storage.close()
    return {"archive_root": str(archive_root), "moved": moved}


def _daemon_loop(cfg: TraderConfig) -> int:
    storage, telegram, sync = _build_runtime(cfg)
    telegram.send("[이벤트] [Trader] 시스템 시작")
    storage.log_event("daemon_start", {"started_at": _now_iso(), "poll_seconds": cfg.poll_seconds})
    last_cycle_ts: Dict[str, float] = {"KR": 0.0, "US": 0.0}
    last_report_date: Dict[str, str] = {"KR": "", "US": ""}

    try:
        while True:
            now = datetime.now(KST)
            date_kst = trade_date_kst()

            if should_run_precompute(cfg, now):
                mark_key = f"precompute_done:{date_kst}"
                if storage.get_meta(mark_key) != "1":
                    result = precompute_all_markets(cfg=cfg, storage=storage, trade_date=date_kst)
                    storage.upsert_meta(mark_key, "1")
                    storage.log_event("precompute_done", result)
                    telegram.send(
                        "[이벤트] [Trader] 일일 패치 완료\n"
                        f"일자={date_kst}\n"
                        f"KR후보={next((r['candidate_count'] for r in result['results'] if r['market']=='KR'), 0)}\n"
                        f"US후보={next((r['candidate_count'] for r in result['results'] if r['market']=='US'), 0)}"
                    )

            if should_run_budget_snapshot(cfg, now):
                if storage.get_day_budget(date_kst) is None:
                    budget = _snapshot_budget(cfg=cfg, storage=storage, sync=sync, trade_date=date_kst)
                    if budget.get("ok"):
                        telegram.send(
                            "[이벤트] [Trader] 일일 예산 스냅샷\n"
                            f"일자={date_kst}\n"
                            f"총현금={float(budget['cash_krw']):.0f}\n"
                            f"1회예산={float(budget['budget_per_trade']):.0f}"
                        )
                    else:
                        telegram.send(
                            "[Action required] [Trader] 예산 스냅샷 실패\n"
                            f"일자={date_kst}\n"
                            f"사유={budget.get('reason')}\n"
                            f"오류={budget.get('errors')}"
                        )

            for market in ("KR", "US"):
                if not cfg.market_enabled(market):
                    continue
                if market == "US":
                    date_market = trade_date_et()
                else:
                    date_market = date_kst

                phase = liquidation_phase(cfg, market)
                if phase:
                    phase_key = f"liq_done:{market}:{date_market}:{phase}"
                    if storage.get_meta(phase_key) != "1":
                        outcome = run_liquidation(
                            cfg=cfg,
                            storage=storage,
                            sync_service=sync,
                            telegram=telegram,
                            market=market,
                            trade_date=date_market,
                            phase=phase,
                        )
                        storage.upsert_meta(phase_key, "1")
                        storage.log_event("liquidation_done", outcome.__dict__)
                        if phase == "final_check" and last_report_date.get(market) != date_market:
                            generate_daily_report(
                                cfg=cfg,
                                storage=storage,
                                telegram=telegram,
                                trade_date=date_market,
                            )
                            last_report_date[market] = date_market
                    continue

                if not is_market_open(market):
                    continue
                now_epoch = time.time()
                if now_epoch - float(last_cycle_ts.get(market, 0.0)) < float(cfg.strategy_cycle_seconds):
                    continue

                # precompute + day budget are mandatory guards for live cycle.
                if storage.get_day_budget(date_kst) is None:
                    continue
                if storage.get_meta(f"precompute_done:{date_kst}") != "1":
                    continue

                cycle_payload = _run_market_cycle(
                    cfg=cfg,
                    storage=storage,
                    sync=sync,
                    telegram=telegram,
                    market=market,
                    trade_date=date_market,
                )
                storage.log_event("strategy_cycle", cycle_payload)
                last_cycle_ts[market] = now_epoch

            storage.log_event("heartbeat", {"ts": _now_iso()})
            time.sleep(max(1, int(cfg.poll_seconds)))
    except KeyboardInterrupt:
        storage.log_event("daemon_stop", {"reason": "keyboard_interrupt", "stopped_at": _now_iso()})
        telegram.send("[이벤트] [Trader] 시스템 종료")
        return 0
    finally:
        storage.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trader pipeline CLI")
    parser.add_argument("--project-root", default=".", help="Project root path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync-account")
    p_sync.add_argument("--market", default="ALL")
    p_sync.add_argument("--strict", action="store_true", default=False)

    p_pre = sub.add_parser("precompute")
    p_pre.add_argument("--market", default="ALL")
    p_pre.add_argument("--date", default=None)

    p_budget = sub.add_parser("snapshot-budget")
    p_budget.add_argument("--date", default=None)

    p_cycle = sub.add_parser("run-cycle")
    p_cycle.add_argument("--market", default="ALL")
    p_cycle.add_argument("--date", default=None)

    p_liq = sub.add_parser("liquidate")
    p_liq.add_argument("--market", default="ALL")
    p_liq.add_argument("--date", default=None)
    p_liq.add_argument("--phase", default="manual")

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--date", default=None)

    p_daemon = sub.add_parser("daemon")
    p_daemon.add_argument("--poll-seconds", type=int, default=None)

    sub.add_parser("archive-reset")
    sub.add_parser("status")

    args = parser.parse_args(argv)
    cfg = load_trader_config(args.project_root)
    if getattr(args, "poll_seconds", None):
        cfg = cfg.__class__(**{**cfg.__dict__, "poll_seconds": int(args.poll_seconds)})

    if args.cmd == "archive-reset":
        result = _archive_and_reset(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    storage, telegram, sync = _build_runtime(cfg)
    try:
        if args.cmd == "status":
            latest = storage.latest_account_snapshot("ALL")
            out = {
                "now": _now_iso(),
                "budget_date_kst": trade_date_kst(),
                "precompute_done": storage.get_meta(f"precompute_done:{trade_date_kst()}"),
                "latest_snapshot": latest,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "sync-account":
            strict = bool(args.strict or cfg.strict_sync)
            result = sync.sync_account(market_scope=str(args.market), strict=strict, trade_date=trade_date_kst())
            payload = result.as_payload()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload.get("ok") else 2

        if args.cmd == "snapshot-budget":
            trade_date = str(args.date or trade_date_kst())
            result = _snapshot_budget(cfg=cfg, storage=storage, sync=sync, trade_date=trade_date)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("ok") else 2

        if args.cmd == "precompute":
            trade_date = str(args.date or trade_date_kst())
            markets = _parse_markets(args.market)
            results = []
            for market in markets:
                if not cfg.market_enabled(market):
                    continue
                results.append(precompute_market(cfg=cfg, storage=storage, market=market, trade_date=trade_date))
            out = {"trade_date": trade_date, "results": results}
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "run-cycle":
            trade_date = str(args.date or trade_date_kst())
            markets = _parse_markets(args.market)
            results = []
            for market in markets:
                if not cfg.market_enabled(market):
                    continue
                results.append(
                    _run_market_cycle(
                        cfg=cfg,
                        storage=storage,
                        sync=sync,
                        telegram=telegram,
                        market=market,
                        trade_date=trade_date_et() if market == "US" else trade_date,
                    )
                )
            print(json.dumps({"trade_date": trade_date, "results": results}, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "liquidate":
            trade_date = str(args.date or trade_date_kst())
            markets = _parse_markets(args.market)
            results = []
            for market in markets:
                if not cfg.market_enabled(market):
                    continue
                results.append(
                    run_liquidation(
                        cfg=cfg,
                        storage=storage,
                        sync_service=sync,
                        telegram=telegram,
                        market=market,
                        trade_date=trade_date_et() if market == "US" else trade_date,
                        phase=str(args.phase),
                    ).__dict__
                )
            print(json.dumps({"trade_date": trade_date, "results": results}, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "report":
            trade_date = str(args.date or trade_date_kst())
            report = generate_daily_report(cfg=cfg, storage=storage, telegram=telegram, trade_date=trade_date)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "daemon":
            storage.close()
            return _daemon_loop(cfg)

        raise RuntimeError(f"unsupported command: {args.cmd}")
    finally:
        try:
            storage.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
