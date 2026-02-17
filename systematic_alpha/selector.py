from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from systematic_alpha.helpers import (
    extract_codes_and_names_from_df,
    first_dict,
    latest_list_of_dict,
    maintained,
    normalize_code,
    normalize_yyyymmdd,
    parse_universe_file,
    pick_first,
    to_float,
)
from systematic_alpha.models import (
    FinalSelection,
    PrevDayStats,
    RealtimeQuality,
    RealtimeStats,
    Stage1Candidate,
    StrategyConfig,
)


class DayTradingSelector:
    def __init__(self, mojito_module, config: StrategyConfig):
        self.mojito = mojito_module
        self.config = config
        self.broker = self.mojito.KoreaInvestment(
            api_key=config.api_key,
            api_secret=config.api_secret,
            acc_no=config.acc_no,
            mock=config.mock,
        )
        self.today_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        self._daily_cache: Dict[str, Optional[PrevDayStats]] = {}
        self._price_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._daily_bars_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_stage1_scan: List[Dict[str, Any]] = []
        self._load_prev_stats_cache()

    def _session_root_dir(self) -> Path:
        if self.config.output_json_path:
            out_path = Path(self.config.output_json_path)
            if out_path.parent.name.lower() == "results":
                return out_path.parent.parent
            return out_path.parent
        if self.config.analytics_dir:
            analytics_path = Path(self.config.analytics_dir)
            if analytics_path.name.lower() == "analytics":
                return analytics_path.parent
            return analytics_path
        return Path("out") / "kr" / self.today_kst

    def _cache_dir(self) -> Path:
        return self._session_root_dir() / "cache"

    def _legacy_liquidity_cache_path(self) -> Path:
        return Path("out") / f"kr_universe_liquidity_{self.today_kst}.csv"

    def _legacy_prev_stats_cache_path(self) -> Path:
        return Path("out") / f"kr_prev_day_stats_{self.today_kst}.csv"

    def _liquidity_cache_path(self) -> Path:
        return self._cache_dir() / "kr_universe_liquidity.csv"

    def _prev_stats_cache_path(self) -> Path:
        return self._cache_dir() / "kr_prev_day_stats.csv"

    def _prev_stats_cache_candidates(self) -> List[Path]:
        paths = [self._prev_stats_cache_path(), self._legacy_prev_stats_cache_path()]
        unique: List[Path] = []
        seen = set()
        for path in paths:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _load_prev_stats_cache(self) -> None:
        for path in self._prev_stats_cache_candidates():
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = normalize_code(row.get("code", "")).strip()
                        if len(code) != 6 or not code.isdigit():
                            continue
                        prev_close = to_float(row.get("prev_close"))
                        prev_volume = to_float(row.get("prev_volume"))
                        prev_turnover = to_float(row.get("prev_turnover"))
                        prev_day_change_pct = to_float(row.get("prev_day_change_pct"))
                        if prev_close is None or prev_close <= 0:
                            continue
                        if prev_volume is None:
                            prev_volume = 0.0
                        if prev_turnover is None:
                            prev_turnover = prev_close * prev_volume
                        self._daily_cache[code] = PrevDayStats(
                            prev_close=prev_close,
                            prev_volume=prev_volume,
                            prev_turnover=prev_turnover,
                            prev_day_change_pct=prev_day_change_pct,
                        )
            except Exception:
                continue

    def _write_prev_stats_cache(self, codes: List[str]) -> None:
        path = self._prev_stats_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["code", "prev_close", "prev_volume", "prev_turnover", "prev_day_change_pct"])
            for code in codes:
                stats = self._daily_cache.get(code)
                if stats is None:
                    continue
                writer.writerow(
                    [
                        code,
                        f"{stats.prev_close:.8f}",
                        f"{stats.prev_volume:.8f}",
                        f"{stats.prev_turnover:.8f}",
                        "" if stats.prev_day_change_pct is None else f"{stats.prev_day_change_pct:.8f}",
                    ]
                )

    def prefetch_prev_day_stats(self, codes: List[str], force_refresh: bool = False) -> Tuple[int, int]:
        unique_codes: List[str] = []
        seen = set()
        for code in codes:
            normalized = normalize_code(code)
            if len(normalized) != 6 or not normalized.isdigit():
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_codes.append(normalized)

        success = 0
        total = len(unique_codes)
        progress_every = max(20, self.config.stage1_log_interval)
        for idx, code in enumerate(unique_codes, start=1):
            if force_refresh and code in self._daily_cache:
                self._daily_cache.pop(code, None)
            stats = self.fetch_prev_day_stats(code)
            if stats is not None:
                success += 1
            if idx % progress_every == 0 or idx == total:
                pct = (idx / total * 100.0) if total > 0 else 100.0
                print(
                    f"[prefetch-prev] scanned={idx}/{total} ({pct:.1f}%), success={success}",
                    flush=True,
                )
            if self.config.rest_sleep_sec > 0:
                time.sleep(self.config.rest_sleep_sec)

        self._write_prev_stats_cache(unique_codes)
        return success, total

    def _read_liquidity_cache(self, path: Path) -> Tuple[List[str], Dict[str, str]]:
        codes: List[str] = []
        names: Dict[str, str] = {}
        if not path.exists():
            return codes, names
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = normalize_code(row.get("code", "")).strip()
                    if len(code) != 6 or not code.isdigit():
                        continue
                    if code in names:
                        continue
                    codes.append(code)
                    name = str(row.get("name", "")).strip()
                    if name:
                        names[code] = name
        except Exception:
            return [], {}
        return codes, names

    def _write_liquidity_cache(self, path: Path, rows: List[Tuple[str, str, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["code", "name", "prev_turnover"])
            for code, name, turnover in rows:
                writer.writerow([code, name, f"{turnover:.0f}"])

    def _build_objective_universe(self) -> Tuple[List[str], Dict[str, str]]:
        cache_path = self._liquidity_cache_path()
        cached_codes, cached_names = self._read_liquidity_cache(cache_path)
        if not cached_codes:
            legacy_path = self._legacy_liquidity_cache_path()
            if legacy_path.exists():
                cached_codes, cached_names = self._read_liquidity_cache(legacy_path)
        if cached_codes:
            target = min(self.config.kr_universe_size, len(cached_codes))
            selected_codes = cached_codes[:target]
            selected_names = {code: cached_names.get(code, "") for code in selected_codes}
            print(
                f"[universe] KR objective pool from cache: {target}/{len(cached_codes)} "
                f"(basis=prev_day_turnover_rank)",
                flush=True,
            )
            return selected_codes, selected_names

        symbols_df = self.broker.fetch_symbols()
        max_count = len(symbols_df.index) if hasattr(symbols_df, "index") else 20000
        all_codes, all_names = extract_codes_and_names_from_df(symbols_df, max_count=max_count)
        scan_budget = min(
            len(all_codes),
            max(50, self.config.max_symbols_scan, self.config.kr_universe_size),
        )
        scan_codes = all_codes[:scan_budget]
        if scan_budget < len(all_codes):
            print(
                f"[universe] KR liquidity scan capped: {scan_budget}/{len(all_codes)} "
                f"(max_symbols_scan={self.config.max_symbols_scan}, kr_universe_size={self.config.kr_universe_size})",
                flush=True,
            )

        ranked: List[Tuple[str, str, float]] = []
        total = len(scan_codes)
        progress_every = max(50, self.config.stage1_log_interval * 10)
        for idx, code in enumerate(scan_codes, start=1):
            try:
                prev = self.fetch_prev_day_stats(code)
                if prev is None or prev.prev_turnover <= 0:
                    continue
                ranked.append((code, all_names.get(code, ""), prev.prev_turnover))
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[universe] liquidity-scan={idx}/{total} ({pct:.1f}%), ranked={len(ranked)}",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        ranked.sort(key=lambda item: item[2], reverse=True)
        if ranked:
            self._write_liquidity_cache(cache_path, ranked)

        target = min(self.config.kr_universe_size, len(ranked))
        selected = ranked[:target]
        selected_codes = [code for code, _, _ in selected]
        selected_names = {code: name for code, name, _ in selected if name}
        print(
            f"[universe] KR objective pool built: {len(selected_codes)}/{len(ranked)} "
            f"(basis=prev_day_turnover_rank)",
            flush=True,
        )
        return selected_codes, selected_names

    def load_universe(self) -> Tuple[List[str], Dict[str, str]]:
        if self.config.universe_file:
            universe_path = Path(self.config.universe_file)
            if not universe_path.exists():
                raise FileNotFoundError(f"Universe file not found: {universe_path}")
            codes, file_names = parse_universe_file(universe_path)
            codes = codes[: self.config.max_symbols_scan]
            names = {code: file_names.get(code, "") for code in codes if file_names.get(code)}

            # Fill missing names from symbol master if available.
            missing_codes = [code for code in codes if code not in names]
            if missing_codes:
                try:
                    symbols_df = self.broker.fetch_symbols()
                    max_count = len(symbols_df.index) if hasattr(symbols_df, "index") else 20000
                    _, all_names = extract_codes_and_names_from_df(symbols_df, max_count=max_count)
                    for code in missing_codes:
                        if all_names.get(code):
                            names[code] = all_names[code]
                except Exception:
                    pass

            return codes, names

        objective_codes, objective_names = self._build_objective_universe()
        if not objective_codes:
            symbols_df = self.broker.fetch_symbols()
            codes, names = extract_codes_and_names_from_df(symbols_df, self.config.max_symbols_scan)
            print(
                f"[universe] fallback to raw symbol list: {len(codes)} (objective pool unavailable)",
                flush=True,
            )
            return codes, names

        final_codes = objective_codes[: self.config.max_symbols_scan]
        final_names = {code: objective_names.get(code, "") for code in final_codes}
        print(
            f"[universe] KR objective pool selected: {len(final_codes)} "
            f"(kr_universe_size={self.config.kr_universe_size}, max_symbols_scan={self.config.max_symbols_scan})",
            flush=True,
        )
        return final_codes, final_names

    def fetch_prev_day_stats(self, code: str) -> Optional[PrevDayStats]:
        if code in self._daily_cache:
            return self._daily_cache[code]

        try:
            resp = self.broker.fetch_ohlcv_recent30(code, timeframe="D", adj_price=True)
        except Exception:
            self._daily_cache[code] = None
            return None

        rows = latest_list_of_dict(resp if isinstance(resp, dict) else {})
        parsed_rows: List[Tuple[str, float, float, float, Optional[float]]] = []
        for row in rows:
            date_key = pick_first(
                row,
                ("stck_bsop_date", "bsop_date", "stck_bsop_dt", "bas_dt", "date", "xymd"),
            )
            date = normalize_yyyymmdd(date_key)
            if not date:
                continue

            close = to_float(pick_first(row, ("stck_clpr", "close", "stck_prpr", "clpr")))
            volume = to_float(pick_first(row, ("acml_vol", "volume", "trade_volume")))
            turnover = to_float(pick_first(row, ("acml_tr_pbmn", "trade_value", "turnover")))
            day_change_pct = to_float(pick_first(row, ("prdy_ctrt", "change_rate")))

            if close is None:
                continue
            if volume is None:
                volume = 0.0
            if turnover is None:
                turnover = close * volume

            parsed_rows.append((date, close, volume, turnover, day_change_pct))

        if not parsed_rows:
            self._daily_cache[code] = None
            return None

        parsed_rows.sort(key=lambda x: x[0], reverse=True)
        past_rows = [row for row in parsed_rows if row[0] < self.today_kst]
        target_rows = past_rows if past_rows else parsed_rows
        if not target_rows:
            self._daily_cache[code] = None
            return None

        prev = target_rows[0]
        prev_prev = target_rows[1] if len(target_rows) > 1 else None
        prev_day_change_pct = prev[4]
        if prev_day_change_pct is None and prev_prev and prev_prev[1] > 0:
            prev_day_change_pct = ((prev[1] - prev_prev[1]) / prev_prev[1]) * 100.0

        stats = PrevDayStats(
            prev_close=prev[1],
            prev_volume=prev[2],
            prev_turnover=prev[3],
            prev_day_change_pct=prev_day_change_pct,
        )
        self._daily_cache[code] = stats
        return stats

    def fetch_price_snapshot(self, code: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        if use_cache and code in self._price_cache:
            return self._price_cache[code]

        try:
            resp = self.broker.fetch_price(code)
        except Exception:
            self._price_cache[code] = None
            return None

        output = first_dict(resp if isinstance(resp, dict) else {})
        if not output:
            self._price_cache[code] = None
            return None

        price = to_float(pick_first(output, ("stck_prpr", "stck_clpr", "last")))
        open_price = to_float(pick_first(output, ("stck_oprc", "open", "oprc")))
        change_pct = to_float(pick_first(output, ("prdy_ctrt", "change_rate", "chg_rt")))
        acml_volume = to_float(pick_first(output, ("acml_vol", "volume")))
        low_price = to_float(pick_first(output, ("stck_lwpr", "low")))
        stock_name_raw = pick_first(output, ("hts_kor_isnm", "prdt_name", "name", "stck_name"))
        stock_name = str(stock_name_raw).strip() if stock_name_raw else ""

        if price is None:
            self._price_cache[code] = None
            return None

        snapshot = {
            "price": price,
            "open": open_price,
            "change_pct": change_pct,
            "acml_volume": acml_volume,
            "low_price": low_price,
            "name": stock_name,
        }
        self._price_cache[code] = snapshot
        return snapshot

    def _pass_change_threshold(self, change_pct: Optional[float], threshold: float) -> bool:
        if change_pct is None:
            return False
        if self.config.long_only:
            return change_pct >= threshold
        return abs(change_pct) >= threshold

    def _pass_gap_threshold(self, gap_pct: Optional[float], threshold: float) -> bool:
        if gap_pct is None:
            return False
        if self.config.long_only:
            return gap_pct >= threshold
        return abs(gap_pct) >= threshold

    def _is_realtime_symbol_eligible(self, realtime: RealtimeStats) -> bool:
        return (
            realtime.execution_ticks >= self.config.min_exec_ticks
            and realtime.orderbook_ticks >= self.config.min_orderbook_ticks
            and realtime.cum_trade_volume >= self.config.min_realtime_cum_volume
        )

    def refresh_candidates_for_decision(self, candidates: List[Stage1Candidate]) -> List[Stage1Candidate]:
        if not candidates:
            return []

        refreshed: List[Stage1Candidate] = []
        total = len(candidates)
        progress_every = max(1, self.config.stage1_log_interval)
        for idx, candidate in enumerate(candidates, start=1):
            try:
                snap = self.fetch_price_snapshot(candidate.code, use_cache=False)
                if snap is None:
                    refreshed.append(candidate)
                    continue

                current_price = snap.get("price")
                if current_price is None:
                    refreshed.append(candidate)
                    continue

                open_price = snap.get("open")
                if open_price is None:
                    open_price = candidate.open_price

                change_pct = snap.get("change_pct")
                if change_pct is None and candidate.prev_close > 0:
                    change_pct = ((current_price - candidate.prev_close) / candidate.prev_close) * 100.0

                gap_pct = candidate.gap_pct
                if candidate.prev_close > 0 and open_price is not None:
                    gap_pct = ((open_price - candidate.prev_close) / candidate.prev_close) * 100.0

                name = candidate.name or str(snap.get("name") or "")
                refreshed.append(
                    Stage1Candidate(
                        code=candidate.code,
                        name=name,
                        current_price=current_price,
                        open_price=open_price if open_price is not None else candidate.open_price,
                        current_change_pct=change_pct if change_pct is not None else candidate.current_change_pct,
                        gap_pct=gap_pct,
                        prev_close=candidate.prev_close,
                        prev_day_volume=candidate.prev_day_volume,
                        prev_day_turnover=candidate.prev_day_turnover,
                    )
                )
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[decision-refresh] refreshed={idx}/{total} ({pct:.1f}%)",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        return refreshed

    def fetch_daily_bars(self, code: str) -> List[Dict[str, Any]]:
        if code in self._daily_bars_cache:
            return self._daily_bars_cache[code]

        try:
            resp = self.broker.fetch_ohlcv_recent30(code, timeframe="D", adj_price=True)
        except Exception:
            self._daily_bars_cache[code] = []
            return []

        rows = latest_list_of_dict(resp if isinstance(resp, dict) else {})
        parsed: List[Dict[str, Any]] = []
        for row in rows:
            date = normalize_yyyymmdd(
                pick_first(row, ("stck_bsop_date", "bsop_date", "stck_bsop_dt", "bas_dt", "date", "xymd"))
            )
            if not date:
                continue
            open_price = to_float(pick_first(row, ("stck_oprc", "open", "oprc")))
            close_price = to_float(pick_first(row, ("stck_clpr", "close", "stck_prpr", "clpr")))
            if close_price is None:
                continue
            parsed.append({"date": date, "open": open_price, "close": close_price})

        parsed.sort(key=lambda item: item["date"])
        self._daily_bars_cache[code] = parsed
        return parsed

    def build_overnight_report_metrics(
        self,
        code: str,
        selection_date: str,
        entry_price: Optional[float],
    ) -> Dict[str, Optional[float | str]]:
        bars = self.fetch_daily_bars(code)
        if not bars:
            return {
                "selection_close": None,
                "next_open": None,
                "next_open_date": None,
                "intraday_return_pct": None,
                "overnight_return_pct": None,
                "total_return_to_next_open_pct": None,
            }

        selection_close: Optional[float] = None
        next_open: Optional[float] = None
        next_open_date: Optional[str] = None

        for idx, bar in enumerate(bars):
            if bar["date"] != selection_date:
                continue
            selection_close = bar.get("close")
            for next_bar in bars[idx + 1 :]:
                if next_bar["date"] > selection_date and next_bar.get("open") is not None:
                    next_open = next_bar["open"]
                    next_open_date = next_bar["date"]
                    break
            break

        intraday_return_pct: Optional[float] = None
        overnight_return_pct: Optional[float] = None
        total_return_to_next_open_pct: Optional[float] = None

        if (
            entry_price is not None
            and entry_price > 0
            and selection_close is not None
        ):
            intraday_return_pct = ((selection_close - entry_price) / entry_price) * 100.0

        if (
            selection_close is not None
            and selection_close > 0
            and next_open is not None
        ):
            overnight_return_pct = ((next_open - selection_close) / selection_close) * 100.0

        if (
            entry_price is not None
            and entry_price > 0
            and next_open is not None
        ):
            total_return_to_next_open_pct = ((next_open - entry_price) / entry_price) * 100.0

        return {
            "selection_close": selection_close,
            "next_open": next_open,
            "next_open_date": next_open_date,
            "intraday_return_pct": intraday_return_pct,
            "overnight_return_pct": overnight_return_pct,
            "total_return_to_next_open_pct": total_return_to_next_open_pct,
        }

    def _build_candidates_with_thresholds(
        self,
        codes: List[str],
        names: Dict[str, str],
        min_change_pct: float,
        min_gap_pct: float,
        min_prev_turnover: float,
        limit: int,
        record_scan: bool = False,
    ) -> List[Stage1Candidate]:
        candidates: List[Stage1Candidate] = []
        stage1_scan_rows: List[Dict[str, Any]] = []
        total = len(codes)
        progress_every = max(1, self.config.stage1_log_interval)
        for idx, code in enumerate(codes, start=1):
            scan_row: Optional[Dict[str, Any]]
            if record_scan:
                scan_row = {
                    "scan_index": idx,
                    "code": code,
                    "name": names.get(code, ""),
                    "current_price": None,
                    "open_price": None,
                    "change_pct": None,
                    "gap_pct": None,
                    "prev_close": None,
                    "prev_day_volume": None,
                    "prev_day_turnover": None,
                    "pass_change": None,
                    "pass_gap": None,
                    "pass_prev_turnover": None,
                    "passed_stage1": False,
                    "skip_reason": "",
                    "min_change_pct": min_change_pct,
                    "min_gap_pct": min_gap_pct,
                    "min_prev_turnover": min_prev_turnover,
                    "long_only": self.config.long_only,
                }
            else:
                scan_row = None

            try:
                snap = self.fetch_price_snapshot(code)
                if snap is None:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "no_price_snapshot"
                        stage1_scan_rows.append(scan_row)
                    continue

                current_price = snap["price"]
                open_price = snap["open"]
                change_pct = snap["change_pct"]
                if scan_row is not None:
                    snap_name = str(snap.get("name") or "").strip()
                    if snap_name and not scan_row.get("name"):
                        scan_row["name"] = snap_name
                    scan_row["current_price"] = current_price
                    scan_row["open_price"] = open_price
                    scan_row["change_pct"] = change_pct
                if current_price is None or open_price is None:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "incomplete_price_snapshot"
                        stage1_scan_rows.append(scan_row)
                    continue

                if change_pct is not None:
                    pass_change = self._pass_change_threshold(change_pct, min_change_pct)
                    if scan_row is not None:
                        scan_row["pass_change"] = pass_change
                    if not pass_change:
                        if scan_row is not None:
                            scan_row["skip_reason"] = "change_threshold"
                            stage1_scan_rows.append(scan_row)
                        continue

                prev = self.fetch_prev_day_stats(code)
                if prev is None or prev.prev_close <= 0:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "no_prev_day_stats"
                        stage1_scan_rows.append(scan_row)
                    continue

                if scan_row is not None:
                    scan_row["prev_close"] = prev.prev_close
                    scan_row["prev_day_volume"] = prev.prev_volume
                    scan_row["prev_day_turnover"] = prev.prev_turnover

                if change_pct is None:
                    change_pct = ((current_price - prev.prev_close) / prev.prev_close) * 100.0
                    if scan_row is not None:
                        scan_row["change_pct"] = change_pct

                pass_change = self._pass_change_threshold(change_pct, min_change_pct)
                if scan_row is not None:
                    scan_row["pass_change"] = pass_change
                if not pass_change:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "change_threshold"
                        stage1_scan_rows.append(scan_row)
                    continue

                gap_pct = ((open_price - prev.prev_close) / prev.prev_close) * 100.0
                pass_gap = self._pass_gap_threshold(gap_pct, min_gap_pct)
                if scan_row is not None:
                    scan_row["gap_pct"] = gap_pct
                    scan_row["pass_gap"] = pass_gap
                if not pass_gap:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "gap_threshold"
                        stage1_scan_rows.append(scan_row)
                    continue

                pass_prev_turnover = prev.prev_turnover >= min_prev_turnover
                if scan_row is not None:
                    scan_row["pass_prev_turnover"] = pass_prev_turnover
                if not pass_prev_turnover:
                    if scan_row is not None:
                        scan_row["skip_reason"] = "prev_turnover_threshold"
                        stage1_scan_rows.append(scan_row)
                    continue

                candidate_name = names.get(code, "") or str(snap.get("name") or "")
                candidates.append(
                    Stage1Candidate(
                        code=code,
                        name=candidate_name,
                        current_price=current_price,
                        open_price=open_price,
                        current_change_pct=change_pct,
                        gap_pct=gap_pct,
                        prev_close=prev.prev_close,
                        prev_day_volume=prev.prev_volume,
                        prev_day_turnover=prev.prev_turnover,
                    )
                )
                if scan_row is not None:
                    scan_row["name"] = candidate_name or scan_row.get("name", "")
                    scan_row["passed_stage1"] = True
                    scan_row["skip_reason"] = ""
                    stage1_scan_rows.append(scan_row)
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[stage1] scanned={idx}/{total} ({pct:.1f}%), candidates={len(candidates)}",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        if self.config.long_only:
            candidates.sort(
                key=lambda c: (c.prev_day_turnover, c.current_change_pct, c.gap_pct),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda c: (c.prev_day_turnover, abs(c.current_change_pct), abs(c.gap_pct)),
                reverse=True,
            )
        if record_scan:
            self.last_stage1_scan = stage1_scan_rows
        return candidates[:limit]

    def build_stage1_candidates(
        self, codes: List[str], names: Dict[str, str]
    ) -> List[Stage1Candidate]:
        return self._build_candidates_with_thresholds(
            codes=codes,
            names=names,
            min_change_pct=self.config.min_change_pct,
            min_gap_pct=self.config.min_gap_pct,
            min_prev_turnover=self.config.min_prev_turnover,
            limit=self.config.pre_candidates,
            record_scan=True,
        )

    def build_fallback_candidates(
        self,
        codes: List[str],
        names: Dict[str, str],
        exclude_codes: set[str],
        needed: int,
    ) -> List[Stage1Candidate]:
        if needed <= 0:
            return []

        fallback_pool_limit = max(self.config.pre_candidates, self.config.final_picks * 10, 50)
        profiles = [
            (
                max(self.config.min_change_pct * 0.7, 0.8),
                max(self.config.min_gap_pct * 0.7, 0.5),
                self.config.min_prev_turnover * 0.5,
            ),
            (
                max(self.config.min_change_pct * 0.4, 0.3),
                max(self.config.min_gap_pct * 0.4, 0.2),
                self.config.min_prev_turnover * 0.2,
            ),
            (0.0, 0.0, 0.0),
        ]

        selected: List[Stage1Candidate] = []
        taken = set(exclude_codes)
        for idx, (chg, gap, turnover) in enumerate(profiles, start=1):
            print(
                f"[fallback] profile={idx} min_change={chg:.2f} min_gap={gap:.2f} min_turnover={turnover:.0f}",
                flush=True,
            )
            profile_candidates = self._build_candidates_with_thresholds(
                codes=codes,
                names=names,
                min_change_pct=chg,
                min_gap_pct=gap,
                min_prev_turnover=turnover,
                limit=fallback_pool_limit,
                record_scan=False,
            )
            added = 0
            for candidate in profile_candidates:
                if candidate.code in taken:
                    continue
                selected.append(candidate)
                taken.add(candidate.code)
                added += 1
                if len(selected) >= needed:
                    return selected
            print(f"[fallback] profile={idx} added={added}", flush=True)

        return selected

    @staticmethod
    def _apply_execution(payload: Dict[str, Any], stats: Dict[str, RealtimeStats]) -> None:
        values = list(payload.values())
        if len(values) < 46:
            return

        code = normalize_code(values[0])
        if code not in stats:
            return

        ref = stats[code]
        ref.got_execution = True
        ref.execution_ticks += 1

        price = to_float(values[2])
        tick_volume = to_float(values[12])
        acml_volume = to_float(values[13])
        strength = to_float(values[18])
        day_low = to_float(values[9])

        if price is not None:
            ref.latest_price = price
        if tick_volume is not None and tick_volume > 0 and price is not None:
            ref.cum_trade_volume += tick_volume
            ref.cum_trade_value += price * tick_volume
        if acml_volume is not None:
            ref.latest_acml_volume = acml_volume
        if strength is not None and strength > 0:
            ref.strength_values.append(strength)
        if day_low is not None:
            if ref.first_reported_low is None:
                ref.first_reported_low = day_low
            elif day_low < ref.first_reported_low:
                ref.low_broken_after_start = True

    @staticmethod
    def _apply_orderbook(payload: Dict[str, Any], stats: Dict[str, RealtimeStats]) -> None:
        values = list(payload.values())
        if len(values) < 45:
            return

        code = normalize_code(values[0])
        if code not in stats:
            return

        ref = stats[code]
        ref.got_orderbook = True
        ref.orderbook_ticks += 1

        total_ask = to_float(values[43])
        total_bid = to_float(values[44])
        if total_ask is None or total_bid is None or total_ask <= 0:
            return
        ref.bid_ask_ratios.append(total_bid / total_ask)

    def collect_realtime(self, codes: List[str]) -> Tuple[Dict[str, RealtimeStats], RealtimeQuality]:
        stats = {code: RealtimeStats() for code in codes}
        if not codes or self.config.collect_seconds <= 0:
            quality = RealtimeQuality(
                realtime_ready=False,
                quality_ok=True,
                coverage_ratio=0.0,
                eligible_count=0,
                total_count=len(codes),
                min_exec_ticks=self.config.min_exec_ticks,
                min_orderbook_ticks=self.config.min_orderbook_ticks,
                min_realtime_cum_volume=self.config.min_realtime_cum_volume,
            )
            return stats, quality

        log_interval = max(1, self.config.realtime_log_interval)
        try:
            ws = self.mojito.KoreaInvestmentWS(
                self.config.api_key,
                self.config.api_secret,
                ["H0STCNT0", "H0STASP0"],
                codes,
                user_id=self.config.user_id,
            )
        except Exception as exc:
            print(f"[realtime] websocket init failed: {exc}", flush=True)
            quality = RealtimeQuality(
                realtime_ready=False,
                quality_ok=False,
                coverage_ratio=0.0,
                eligible_count=0,
                total_count=len(codes),
                min_exec_ticks=self.config.min_exec_ticks,
                min_orderbook_ticks=self.config.min_orderbook_ticks,
                min_realtime_cum_volume=self.config.min_realtime_cum_volume,
            )
            return stats, quality

        print(
            f"[realtime] starting websocket: codes={len(codes)}, duration={self.config.collect_seconds}s, heartbeat={log_interval}s",
            flush=True,
        )
        try:
            ws.start()
        except Exception as exc:
            print(f"[realtime] websocket start failed: {exc}", flush=True)
            quality = RealtimeQuality(
                realtime_ready=False,
                quality_ok=False,
                coverage_ratio=0.0,
                eligible_count=0,
                total_count=len(codes),
                min_exec_ticks=self.config.min_exec_ticks,
                min_orderbook_ticks=self.config.min_orderbook_ticks,
                min_realtime_cum_volume=self.config.min_realtime_cum_volume,
            )
            return stats, quality
        execution_events = 0
        orderbook_events = 0
        first_exec_logged = False
        first_orderbook_logged = False
        started = time.time()
        deadline = started + self.config.collect_seconds
        next_log = started + log_interval

        try:
            while time.time() < deadline:
                if time.time() >= next_log:
                    remain = max(int(deadline - time.time()), 0)
                    exec_symbols = sum(1 for ref in stats.values() if ref.got_execution)
                    ob_symbols = sum(1 for ref in stats.values() if ref.got_orderbook)
                    queue_size: str = "n/a"
                    try:
                        queue_size = str(ws.queue.qsize())
                    except Exception:
                        pass
                    print(
                        "[realtime] "
                        f"remain={remain}s, "
                        f"exec_events={execution_events}, orderbook_events={orderbook_events}, "
                        f"exec_symbols={exec_symbols}/{len(codes)}, orderbook_symbols={ob_symbols}/{len(codes)}, "
                        f"queue={queue_size}",
                        flush=True,
                    )
                    next_log += log_interval

                try:
                    event = ws.queue.get(timeout=1.0)
                except Empty:
                    continue
                except Exception:
                    continue

                if not isinstance(event, (list, tuple)) or len(event) < 2:
                    continue
                payload = event[1]
                if not isinstance(payload, dict):
                    continue

                values_len = len(payload)
                if 46 <= values_len < 55:
                    self._apply_execution(payload, stats)
                    execution_events += 1
                    if not first_exec_logged and execution_events > 0:
                        print("[realtime] first execution tick received.", flush=True)
                        first_exec_logged = True
                elif values_len >= 55:
                    self._apply_orderbook(payload, stats)
                    orderbook_events += 1
                    if not first_orderbook_logged and orderbook_events > 0:
                        print("[realtime] first orderbook tick received.", flush=True)
                        first_orderbook_logged = True
        finally:
            try:
                ws.terminate()
            finally:
                try:
                    ws.join(timeout=3)
                except Exception:
                    pass

        exec_symbols = sum(1 for ref in stats.values() if ref.got_execution)
        ob_symbols = sum(1 for ref in stats.values() if ref.got_orderbook)
        eligible_count = sum(1 for ref in stats.values() if self._is_realtime_symbol_eligible(ref))
        total_count = len(codes)
        coverage_ratio = (eligible_count / total_count) if total_count > 0 else 0.0
        realtime_ready = execution_events > 0 and orderbook_events > 0
        quality_ok = coverage_ratio >= self.config.min_realtime_coverage_ratio
        print(
            "[realtime] finished: "
            f"exec_events={execution_events}, orderbook_events={orderbook_events}, "
            f"exec_symbols={exec_symbols}/{len(codes)}, orderbook_symbols={ob_symbols}/{len(codes)}, "
            f"eligible={eligible_count}/{total_count}, coverage={coverage_ratio:.3f}",
            flush=True,
        )
        print(
            "[realtime] quality gate: "
            f"min_exec_ticks={self.config.min_exec_ticks}, "
            f"min_orderbook_ticks={self.config.min_orderbook_ticks}, "
            f"min_cum_volume={self.config.min_realtime_cum_volume}, "
            f"min_coverage_ratio={self.config.min_realtime_coverage_ratio:.3f}, "
            f"quality_ok={quality_ok}",
            flush=True,
        )

        quality = RealtimeQuality(
            realtime_ready=realtime_ready,
            quality_ok=quality_ok,
            coverage_ratio=coverage_ratio,
            eligible_count=eligible_count,
            total_count=total_count,
            min_exec_ticks=self.config.min_exec_ticks,
            min_orderbook_ticks=self.config.min_orderbook_ticks,
            min_realtime_cum_volume=self.config.min_realtime_cum_volume,
        )
        return stats, quality

    def evaluate(
        self,
        candidates: List[Stage1Candidate],
        stats: Dict[str, RealtimeStats],
        realtime_ready: bool,
    ) -> List[FinalSelection]:
        results: List[FinalSelection] = []

        for candidate in candidates:
            realtime = stats.get(candidate.code, RealtimeStats())
            symbol_realtime_eligible = self._is_realtime_symbol_eligible(realtime)
            vwap = (
                realtime.cum_trade_value / realtime.cum_trade_volume
                if realtime.cum_trade_volume > 0
                else None
            )
            volume_ratio = (
                realtime.latest_acml_volume / candidate.prev_day_volume
                if realtime.latest_acml_volume is not None and candidate.prev_day_volume > 0
                else None
            )
            strength_ok, strength_avg, strength_hit_ratio = maintained(
                realtime.strength_values,
                self.config.min_strength,
                self.config.min_strength_samples,
                self.config.min_maintain_ratio,
            )
            bid_ask_ok, bid_ask_avg, bid_ask_hit_ratio = maintained(
                realtime.bid_ask_ratios,
                self.config.min_bid_ask_ratio,
                self.config.min_bid_ask_samples,
                self.config.min_maintain_ratio,
            )
            current_vs_vwap = (
                realtime.latest_price is not None
                and vwap is not None
                and realtime.latest_price >= vwap
            )

            if realtime_ready:
                conditions = {
                    "change_pct": self._pass_change_threshold(
                        candidate.current_change_pct, self.config.min_change_pct
                    ),
                    "gap_pct": self._pass_gap_threshold(candidate.gap_pct, self.config.min_gap_pct),
                    "prev_turnover": candidate.prev_day_turnover >= self.config.min_prev_turnover,
                    "strength_maintained": symbol_realtime_eligible and strength_ok,
                    "volume_ratio": symbol_realtime_eligible
                    and volume_ratio is not None
                    and volume_ratio >= self.config.min_vol_ratio,
                    "bid_ask_maintained": symbol_realtime_eligible and bid_ask_ok,
                    "price_above_vwap": symbol_realtime_eligible and current_vs_vwap,
                    "low_not_broken": symbol_realtime_eligible and (not realtime.low_broken_after_start),
                }
                max_score = 8
                pass_cut = self.config.min_pass_conditions
            else:
                conditions = {
                    "change_pct": self._pass_change_threshold(
                        candidate.current_change_pct, self.config.min_change_pct
                    ),
                    "gap_pct": self._pass_gap_threshold(candidate.gap_pct, self.config.min_gap_pct),
                    "prev_turnover": candidate.prev_day_turnover >= self.config.min_prev_turnover,
                }
                max_score = 3
                pass_cut = 3

            score = sum(1 for value in conditions.values() if value)
            metrics = {
                "current_change_pct": candidate.current_change_pct,
                "gap_pct": candidate.gap_pct,
                "prev_day_turnover": candidate.prev_day_turnover,
                "prev_day_volume": candidate.prev_day_volume,
                "strength_avg": strength_avg,
                "strength_hit_ratio": strength_hit_ratio,
                "bid_ask_avg": bid_ask_avg,
                "bid_ask_hit_ratio": bid_ask_hit_ratio,
                "volume_ratio": volume_ratio,
                "vwap": vwap,
                "execution_ticks": float(realtime.execution_ticks),
                "orderbook_ticks": float(realtime.orderbook_ticks),
                "realtime_eligible": 1.0 if symbol_realtime_eligible else 0.0,
                "latest_price": (
                    realtime.latest_price if realtime.latest_price is not None else candidate.current_price
                ),
            }

            results.append(
                FinalSelection(
                    code=candidate.code,
                    name=candidate.name,
                    score=score,
                    max_score=max_score,
                    passed=score >= pass_cut,
                    conditions=conditions,
                    metrics=metrics,
                )
            )

        results.sort(
            key=lambda item: (
                item.score,
                item.metrics.get("strength_avg") or 0.0,
                item.metrics.get("volume_ratio") or 0.0,
                item.metrics.get("bid_ask_avg") or 0.0,
                item.metrics.get("prev_day_turnover") or 0.0,
            ),
            reverse=True,
        )
        return results
