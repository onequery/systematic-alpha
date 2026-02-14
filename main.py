from __future__ import annotations

import argparse
import os

from systematic_alpha.cli import run
from systematic_alpha.credentials import load_credentials
from systematic_alpha.dotenv import load_dotenv
from systematic_alpha.helpers import env_float, env_int
from systematic_alpha.models import StrategyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KIS intraday scalping selector (REST + WebSocket, mojito2)."
    )
    parser.add_argument("--key-file", type=str, default=None, help="Path to KIS key file.")
    parser.add_argument(
        "--universe-file",
        type=str,
        default=os.getenv("UNIVERSE_CODES_FILE"),
        help="Optional text/csv file containing 6-digit stock codes.",
    )
    parser.add_argument(
        "--collect-seconds",
        type=int,
        default=env_int("COLLECT_SECONDS", 600),
        help="Realtime collection duration in seconds.",
    )
    parser.add_argument(
        "--max-symbols-scan",
        type=int,
        default=env_int("MAX_SYMBOLS_SCAN", 400),
        help="Maximum symbol count to scan in stage1.",
    )
    parser.add_argument(
        "--pre-candidates",
        type=int,
        default=env_int("PRE_CANDIDATES", 40),
        help="Max candidate count after stage1.",
    )
    parser.add_argument(
        "--final-picks",
        type=int,
        default=env_int("FINAL_PICKS", 3),
        help="Number of final picks.",
    )
    parser.add_argument(
        "--min-change-pct",
        type=float,
        default=env_float("MIN_CHANGE_PCT", 3.0),
        help="Absolute current day change %% threshold.",
    )
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=env_float("MIN_GAP_PCT", 2.0),
        help="Absolute opening gap %% threshold.",
    )
    parser.add_argument(
        "--min-prev-turnover",
        type=float,
        default=env_float("MIN_PREV_TURNOVER", 10_000_000_000),
        help="Previous day turnover threshold (KRW). Default is 10,000,000,000.",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=env_float("MIN_STRENGTH", 100.0),
        help="Execution strength threshold.",
    )
    parser.add_argument(
        "--min-vol-ratio",
        type=float,
        default=env_float("MIN_VOL_RATIO", 0.10),
        help="Intraday volume ratio threshold (today volume / prev day volume).",
    )
    parser.add_argument(
        "--min-bid-ask-ratio",
        type=float,
        default=env_float("MIN_BID_ASK_RATIO", 1.2),
        help="Bid/ask remaining quantity ratio threshold.",
    )
    parser.add_argument(
        "--min-pass-conditions",
        type=int,
        default=env_int("MIN_PASS_CONDITIONS", 5),
        help="Pass cut for 8-condition realtime mode.",
    )
    parser.add_argument(
        "--min-maintain-ratio",
        type=float,
        default=env_float("MIN_MAINTAIN_RATIO", 0.6),
        help="Required hit ratio for 'maintained' conditions.",
    )
    parser.add_argument(
        "--rest-sleep",
        type=float,
        default=env_float("REST_SLEEP_SEC", 0.03),
        help="Small delay between REST calls to reduce burst rate.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=os.getenv("KIS_MOCK", "0") == "1",
        help="Use KIS mock server for REST calls.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=os.getenv("KIS_USER_ID"),
        help="Optional user id for websocket notice subscription.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.getenv("SELECTION_OUTPUT_JSON"),
        help="Optional output json file path.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StrategyConfig:
    api_key, api_secret, acc_no, file_user_id = load_credentials(args.key_file)
    user_id = args.user_id or file_user_id
    return StrategyConfig(
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=args.mock,
        universe_file=args.universe_file,
        max_symbols_scan=args.max_symbols_scan,
        pre_candidates=args.pre_candidates,
        final_picks=args.final_picks,
        collect_seconds=args.collect_seconds,
        rest_sleep_sec=args.rest_sleep,
        min_change_pct=args.min_change_pct,
        min_gap_pct=args.min_gap_pct,
        min_prev_turnover=args.min_prev_turnover,
        min_strength=args.min_strength,
        min_vol_ratio=args.min_vol_ratio,
        min_bid_ask_ratio=args.min_bid_ask_ratio,
        min_pass_conditions=args.min_pass_conditions,
        min_maintain_ratio=args.min_maintain_ratio,
        min_strength_samples=3,
        min_bid_ask_samples=3,
        output_json_path=args.output_json,
    )


def main() -> None:
    load_dotenv(".env", override=False)
    args = parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()
