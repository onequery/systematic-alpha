#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def _normalize_profile(raw: str) -> str:
    p = (raw or "prod").strip().lower()
    if p in {"", "prod", "main", "default"}:
        return "prod"
    return p


def _bucket_path(root: Path, bucket: str, profile: str) -> Path:
    if profile == "prod":
        return root / bucket / "trader"
    return root / bucket / f"trader_{profile}"


def _rows(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    cur = con.execute(sql, params)
    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        out.append(dict(r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick trader DB summary")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--profile", default="prod")
    ap.add_argument("--date", default="")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    profile = _normalize_profile(args.profile)
    state_dir = _bucket_path(root, "state", profile)
    db_path = state_dir / "trader.sqlite"

    if not db_path.exists():
        print(json.dumps({"ok": False, "error": "db_not_found", "db_path": str(db_path)}, ensure_ascii=False, indent=2))
        return 2

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    where = ""
    params: tuple[Any, ...] = ()
    if args.date:
        where = " WHERE trade_date = ?"
        params = (str(args.date),)

    precompute = _rows(
        con,
        (
            "SELECT premarket_calc_id, trade_date, market, status, candidate_count, started_at, finished_at, detail_json "
            "FROM premarket_calc_log"
            f"{where} "
            "ORDER BY premarket_calc_id DESC LIMIT ?"
        ),
        params + (int(args.limit),),
    )

    candidates = _rows(
        con,
        (
            "SELECT trade_date, market, COUNT(*) AS candidate_count "
            "FROM symbol_plan WHERE is_candidate = 1"
            + (" AND trade_date = ?" if args.date else "")
            + " GROUP BY trade_date, market ORDER BY trade_date DESC, market"
        ),
        params,
    )

    filters = _rows(
        con,
        (
            "SELECT trade_date, market, index_symbol, prev_close, ma20_prev, trading_enabled, reason, computed_at "
            "FROM market_filter"
            f"{where} ORDER BY trade_date DESC, market LIMIT ?"
        ),
        params + (int(args.limit),),
    )

    budget = _rows(
        con,
        (
            "SELECT trade_date, day_start_cash_snapshot_total, per_trade_ratio, budget_per_trade, captured_at "
            "FROM day_budget"
            f"{where} ORDER BY trade_date DESC LIMIT ?"
        ),
        params + (int(args.limit),),
    )

    out = {
        "ok": True,
        "profile": profile,
        "db_path": str(db_path),
        "date_filter": str(args.date or ""),
        "precompute": [
            {
                **{k: v for k, v in row.items() if k != "detail_json"},
                "detail": (json.loads(row.get("detail_json") or "{}") if row.get("detail_json") else {}),
            }
            for row in precompute
        ],
        "candidate_counts": candidates,
        "market_filters": [
            {**row, "trading_enabled": bool(int(row.get("trading_enabled", 0) or 0))}
            for row in filters
        ],
        "day_budget": budget,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
