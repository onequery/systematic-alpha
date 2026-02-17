from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import requests

DEFAULT_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"


def normalize_symbol(raw: object) -> str:
    if raw is None:
        return ""
    text = str(raw).strip().upper()
    if not text:
        return ""
    return text


def parse_sp500_csv(text: str) -> Tuple[List[str], Dict[str, str]]:
    symbols: List[str] = []
    names: Dict[str, str] = {}
    seen = set()
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        raw_symbol = (
            row.get("symbol")
            or row.get("Symbol")
            or row.get("ticker")
            or row.get("Ticker")
            or ""
        )
        symbol = normalize_symbol(raw_symbol)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)

        raw_name = (
            row.get("name")
            or row.get("Name")
            or row.get("security")
            or row.get("Security")
            or ""
        )
        name = str(raw_name).strip()
        if name:
            names[symbol] = name
    return symbols, names


def write_universe_cache(path: Path, symbols: List[str], names: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "name"])
        for symbol in symbols:
            writer.writerow([symbol, names.get(symbol, "")])


def load_bundled_snapshot(project_root: Path) -> Tuple[List[str], Dict[str, str]]:
    snapshot_path = project_root / "systematic_alpha" / "data" / "us_sp500_snapshot.csv"
    if not snapshot_path.exists():
        return [], {}
    try:
        text = snapshot_path.read_text(encoding="utf-8")
    except Exception:
        return [], {}
    return parse_sp500_csv(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch and cache US objective universe (S&P 500 constituents)."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--source-url", type=str, default=DEFAULT_SOURCE_URL)
    parser.add_argument("--min-count", type=int, default=450)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "out"
    today_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    out_path = out_dir / today_kst / "us" / "cache" / "us_sp500_constituents.csv"
    min_count = max(1, args.min_count)

    print(f"[prefetch] source={args.source_url}", flush=True)
    print(f"[prefetch] target={out_path}", flush=True)

    symbols: List[str] = []
    names: Dict[str, str] = {}
    source_label = args.source_url
    try:
        resp = requests.get(args.source_url, timeout=10)
        resp.raise_for_status()
        symbols, names = parse_sp500_csv(resp.text)
    except Exception as exc:
        print(f"[prefetch] remote failed: {exc}", flush=True)
        symbols, names = load_bundled_snapshot(project_root)
        source_label = "bundled_snapshot"
        if not symbols:
            print("[prefetch] failed: bundled snapshot unavailable", flush=True)
            return 1

    if len(symbols) < min_count:
        print(
            f"[prefetch] failed: symbol count too small ({len(symbols)} < {min_count})",
            flush=True,
        )
        return 2

    write_universe_cache(out_path, symbols, names)
    print(
        f"[prefetch] success: source={source_label}, symbols={len(symbols)} saved={out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
