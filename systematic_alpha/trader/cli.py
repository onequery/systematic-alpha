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
    budget_snapshot_time,
    is_market_open,
    liquidation_phase,
    precompute_window,
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
        "state": cfg.state_dir,
        "out": cfg.out_dir,
        "logs": cfg.logs_dir,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _build_runtime(cfg: TraderConfig):
    _ensure_trader_paths(cfg)
    db_path = cfg.state_dir / "trader.sqlite"
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


def _analyze_precompute_result(result: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    api_failures: List[Dict[str, object]] = []
    bar_shortages: List[Dict[str, object]] = []
    for row in list(result.get("results", []) if isinstance(result, dict) else []):
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "") or "").upper()
        status = str(row.get("status", "") or "").upper()
        detail = row.get("detail", {}) if isinstance(row.get("detail"), dict) else {}
        if status != "OK":
            api_failures.append(
                {
                    "market": market,
                    "kind": "precompute_error",
                    "error": str(detail.get("error", "unknown_precompute_error")),
                }
            )
            continue

        index_filter = detail.get("index_filter", {}) if isinstance(detail.get("index_filter"), dict) else {}
        reason = str(index_filter.get("reason", "") or "")
        if reason == "index_fetch_failed":
            api_failures.append(
                {
                    "market": market,
                    "kind": "index_fetch_failed",
                    "error": index_filter.get("fetch_diagnostics", {}),
                }
            )
        elif reason == "insufficient_index_bars":
            bar_shortages.append(
                {
                    "market": market,
                    "kind": "insufficient_index_bars",
                    "bars": int(index_filter.get("bars", 0) or 0),
                    "required_bars": int(index_filter.get("required_bars", 0) or 0),
                    "index_symbol": str(index_filter.get("index_symbol", "") or ""),
                }
            )

    return {
        "api_failures": api_failures,
        "bar_shortages": bar_shortages,
    }


def _notify_precompute_result(
    *,
    storage: TraderStorage,
    telegram: TelegramClient,
    trade_date: str,
    trigger: str,
    result: Dict[str, object],
) -> None:
    rows = list(result.get("results", []) if isinstance(result, dict) else [])
    kr_candidates = next(
        (int(x.get("candidate_count", 0) or 0) for x in rows if isinstance(x, dict) and x.get("market") == "KR"),
        0,
    )
    us_candidates = next(
        (int(x.get("candidate_count", 0) or 0) for x in rows if isinstance(x, dict) and x.get("market") == "US"),
        0,
    )
    telegram.send(
        "[이벤트] [Trader] 일일 패치 완료\n"
        f"일자={trade_date}\n"
        f"트리거={trigger}\n"
        f"KR후보={kr_candidates}\n"
        f"US후보={us_candidates}"
    )

    analyzed = _analyze_precompute_result(result)
    api_failures = analyzed["api_failures"]
    bar_shortages = analyzed["bar_shortages"]

    if api_failures:
        storage.log_event(
            "precompute_api_failure_alert",
            {"trade_date": trade_date, "trigger": trigger, "failures": api_failures},
        )
        lines = []
        for item in api_failures[:3]:
            mk = str(item.get("market", "") or "")
            kind = str(item.get("kind", "") or "")
            err = str(item.get("error", ""))[:220]
            lines.append(f"- {mk} ({kind}): {err}")
        telegram.send(
            "[Action required] [Trader] 사전계산 API 실패\n"
            f"일자={trade_date}\n"
            f"건수={len(api_failures)}\n"
            "상세(최대3건):\n"
            + "\n".join(lines)
        )

    if bar_shortages:
        storage.log_event(
            "precompute_index_bar_shortage_notice",
            {"trade_date": trade_date, "trigger": trigger, "shortages": bar_shortages},
        )
        lines = []
        for item in bar_shortages[:3]:
            lines.append(
                f"- {item.get('market')} {item.get('index_symbol')}: bars={item.get('bars')} / need={item.get('required_bars')}"
            )
        telegram.send(
            "[이벤트] [Trader] 지수 바 부족\n"
            f"일자={trade_date}\n"
            f"건수={len(bar_shortages)}\n"
            "상세(최대3건):\n"
            + "\n".join(lines)
        )


def _ensure_daily_bootstrap(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    sync: AccountSyncService,
    telegram: TelegramClient,
    now_kst_dt: datetime,
    trade_date: str,
) -> Dict[str, object]:
    precompute_key = f"precompute_done:{trade_date}"
    precompute_done = str(storage.get_meta(precompute_key, "0")) == "1"
    budget_done = storage.get_day_budget(trade_date) is not None

    pre_start, pre_end = precompute_window(cfg, now_kst_dt)
    snap_time = budget_snapshot_time(cfg, now_kst_dt)
    any_market_open = bool(is_market_open("KR") or is_market_open("US"))

    trigger = ""
    if any_market_open:
        trigger = "market_open"
    elif now_kst_dt >= pre_end:
        trigger = "after_precompute_window"
    elif now_kst_dt >= snap_time:
        trigger = "after_budget_snapshot_time"

    ran_precompute = False
    ran_budget_snapshot = False

    if (not precompute_done) and (any_market_open or now_kst_dt >= pre_end):
        result = precompute_all_markets(cfg=cfg, storage=storage, trade_date=trade_date)
        storage.upsert_meta(precompute_key, "1")
        storage.log_event(
            "precompute_catchup_done",
            {"trade_date": trade_date, "trigger": trigger or "catchup", "result": result},
        )
        _notify_precompute_result(
            storage=storage,
            telegram=telegram,
            trade_date=trade_date,
            trigger=f"catchup:{trigger or 'catchup'}",
            result=result,
        )
        ran_precompute = True

    if (not budget_done) and (any_market_open or now_kst_dt >= snap_time):
        budget = _snapshot_budget(cfg=cfg, storage=storage, sync=sync, trade_date=trade_date)
        if budget.get("ok"):
            storage.log_event(
                "budget_snapshot_catchup",
                {
                    "trade_date": trade_date,
                    "trigger": trigger or "catchup",
                    "cash_krw": float(budget.get("cash_krw", 0.0) or 0.0),
                    "budget_per_trade": float(budget.get("budget_per_trade", 0.0) or 0.0),
                },
            )
            telegram.send(
                "[이벤트] [Trader] 일일 예산 지연 보정\n"
                f"일자={trade_date}\n"
                f"총현금={float(budget.get('cash_krw', 0.0) or 0.0):.0f}\n"
                f"1회예산={float(budget.get('budget_per_trade', 0.0) or 0.0):.0f}"
            )
        else:
            storage.log_event(
                "budget_snapshot_catchup_failed",
                {
                    "trade_date": trade_date,
                    "trigger": trigger or "catchup",
                    "reason": budget.get("reason"),
                    "errors": budget.get("errors", []),
                },
            )
            telegram.send(
                "[Action required] [Trader] 예산 스냅샷 지연 보정 실패\n"
                f"일자={trade_date}\n"
                f"사유={budget.get('reason')}\n"
                f"오류={budget.get('errors')}"
            )
        ran_budget_snapshot = True

    return {
        "trade_date": trade_date,
        "trigger": trigger or "none",
        "ran_precompute": ran_precompute,
        "ran_budget_snapshot": ran_budget_snapshot,
    }


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
    ]
    for bucket in ("state", "out", "logs"):
        base = cfg.project_root / bucket
        if not base.exists():
            continue
        for path in base.glob("trader*"):
            move_targets.append(path)

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
    storage = TraderStorage(cfg.state_dir / "trader.sqlite")
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

            catchup = _ensure_daily_bootstrap(
                cfg=cfg,
                storage=storage,
                sync=sync,
                telegram=telegram,
                now_kst_dt=now,
                trade_date=date_kst,
            )
            if catchup.get("ran_precompute") or catchup.get("ran_budget_snapshot"):
                storage.log_event("daily_bootstrap_catchup", catchup)

            if should_run_precompute(cfg, now):
                mark_key = f"precompute_done:{date_kst}"
                if storage.get_meta(mark_key) != "1":
                    result = precompute_all_markets(cfg=cfg, storage=storage, trade_date=date_kst)
                    storage.upsert_meta(mark_key, "1")
                    storage.log_event("precompute_done", result)
                    _notify_precompute_result(
                        storage=storage,
                        telegram=telegram,
                        trade_date=date_kst,
                        trigger="scheduled_precompute",
                        result=result,
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
