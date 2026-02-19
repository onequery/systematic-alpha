from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from systematic_alpha.dotenv import load_dotenv
from systematic_alpha.agent_lab.auto_strategy import run_auto_strategy_daemon
from systematic_alpha.agent_lab.orchestrator import AgentLabOrchestrator
from systematic_alpha.agent_lab.self_heal import run_data_reception_self_heal
from systematic_alpha.agent_lab.telegram_chat import run_telegram_chat_worker
from systematic_alpha.network_env import apply_network_env_guard


def _echo(payload: Dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Lab for multi-agent strategy research and paper-trading workflow."
    )
    parser.add_argument("--project-root", type=str, default=".")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--capital-krw", type=float, required=True)
    p_init.add_argument("--agents", type=int, default=3)

    p_ingest = sub.add_parser("ingest-session")
    p_ingest.add_argument("--market", type=str, choices=["KR", "US", "kr", "us"], required=True)
    p_ingest.add_argument("--date", type=str, required=True)

    p_prop = sub.add_parser("propose-orders")
    p_prop.add_argument("--market", type=str, choices=["KR", "US", "kr", "us"], required=True)
    p_prop.add_argument("--date", type=str, required=True)

    p_daily = sub.add_parser("daily-review")
    p_daily.add_argument("--date", type=str, required=True)

    p_weekly = sub.add_parser("weekly-council")
    p_weekly.add_argument("--week", type=str, required=True, help="format: YYYY-Www")

    p_report = sub.add_parser("report")
    p_report.add_argument("--from", dest="date_from", type=str, required=True)
    p_report.add_argument("--to", dest="date_to", type=str, required=True)

    p_preopen = sub.add_parser("preopen-plan")
    p_preopen.add_argument("--market", type=str, choices=["KR", "US", "kr", "us"], required=True)
    p_preopen.add_argument("--date", type=str, required=True)

    p_close = sub.add_parser("close-report")
    p_close.add_argument("--market", type=str, choices=["KR", "US", "kr", "us"], required=True)
    p_close.add_argument("--date", type=str, required=True)

    p_sanitize = sub.add_parser("sanitize-state")
    p_sanitize.add_argument("--skip-pending-cleanup", action="store_true")
    p_sanitize.add_argument("--skip-runtime-cleanup", action="store_true")
    p_sanitize.add_argument("--retain-days", type=int, default=30)
    p_sanitize.add_argument("--keep-agent-memories", type=int, default=300)

    p_chat = sub.add_parser("telegram-chat")
    p_chat.add_argument("--poll-timeout", type=int, default=25)
    p_chat.add_argument("--idle-sleep", type=float, default=1.0)
    p_chat.add_argument("--memory-limit", type=int, default=20)
    p_chat.add_argument("--once", action="store_true")

    p_auto = sub.add_parser("auto-strategy-daemon")
    p_auto.add_argument("--poll-seconds", type=int, default=30)
    p_auto.add_argument("--cooldown-minutes", type=int, default=180)
    p_auto.add_argument("--max-updates-per-day", type=int, default=2)
    p_auto.add_argument("--once", action="store_true")

    p_heal = sub.add_parser("self-heal")
    p_heal.add_argument("--market", type=str, choices=["KR", "US", "kr", "us"], required=True)
    p_heal.add_argument("--date", type=str, required=True)
    p_heal.add_argument("--log-path", type=str, default="")
    p_heal.add_argument("--output-json", type=str, default="")
    p_heal.add_argument("--failure-tail", type=str, default="")
    p_heal.add_argument("--no-auto-apply", action="store_true")

    return parser.parse_args()


def main() -> None:
    load_dotenv(".env", override=False)
    # self-heal:network-guard-v1
    apply_network_env_guard()
    args = parse_args()
    if args.command == "telegram-chat":
        payload = run_telegram_chat_worker(
            project_root=Path(args.project_root),
            poll_timeout=args.poll_timeout,
            idle_sleep=args.idle_sleep,
            memory_limit=args.memory_limit,
            once=bool(args.once),
        )
        _echo(payload)
        return
    if args.command == "self-heal":
        payload = run_data_reception_self_heal(
            project_root=Path(args.project_root),
            market=str(args.market).upper(),
            run_date=args.date,
            log_path=str(args.log_path or "").strip() or None,
            output_json_path=str(args.output_json or "").strip() or None,
            failure_tail=args.failure_tail or "",
            auto_apply=not bool(args.no_auto_apply),
        )
        _echo(payload)
        return
    if args.command == "auto-strategy-daemon":
        payload = run_auto_strategy_daemon(
            project_root=Path(args.project_root),
            poll_seconds=args.poll_seconds,
            cooldown_minutes=args.cooldown_minutes,
            max_updates_per_day=args.max_updates_per_day,
            once=bool(args.once),
        )
        _echo(payload)
        return

    orchestrator = AgentLabOrchestrator(project_root=Path(args.project_root))
    try:
        cmd = args.command
        if cmd == "init":
            payload = orchestrator.init_lab(capital_krw=args.capital_krw, agents=args.agents)
        elif cmd == "ingest-session":
            payload = orchestrator.ingest_session(market=str(args.market).upper(), yyyymmdd=args.date)
        elif cmd == "propose-orders":
            payload = orchestrator.propose_orders(
                market=str(args.market).upper(),
                yyyymmdd=args.date,
            )
        elif cmd == "daily-review":
            payload = orchestrator.daily_review(yyyymmdd=args.date)
        elif cmd == "weekly-council":
            payload = orchestrator.weekly_council(week_id=args.week)
        elif cmd == "report":
            payload = orchestrator.report(date_from=args.date_from, date_to=args.date_to)
        elif cmd == "preopen-plan":
            payload = orchestrator.preopen_plan_report(
                market=str(args.market).upper(),
                yyyymmdd=args.date,
            )
        elif cmd == "close-report":
            payload = orchestrator.session_close_report(
                market=str(args.market).upper(),
                yyyymmdd=args.date,
            )
        elif cmd == "sanitize-state":
            payload = orchestrator.sanitize_legacy_constraints(
                clean_pending_proposals=not bool(args.skip_pending_cleanup),
                cleanup_runtime_state=not bool(args.skip_runtime_cleanup),
                retain_days=int(args.retain_days),
                keep_agent_memories=int(args.keep_agent_memories),
            )
        else:
            raise RuntimeError(f"unsupported command: {cmd}")
        _echo(payload)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
