from __future__ import annotations

import csv
import os
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
        self._api_diag_counts: Dict[str, int] = {}
        self._api_diag_samples: List[Dict[str, str]] = []
        self._api_diag_last_by_code: Dict[str, str] = {}
        self._rate_limit_retries = max(
            0, int(float(os.getenv("TRADER_RATE_LIMIT_RETRIES", "3") or 3))
        )
        self._rate_limit_backoff_sec = max(
            0.1, float(os.getenv("TRADER_RATE_LIMIT_BACKOFF_SEC", "2.0") or 2.0)
        )
        self._rate_limit_backoff_max_sec = max(
            0.1, float(os.getenv("TRADER_RATE_LIMIT_BACKOFF_MAX_SEC", "20.0") or 20.0)
        )
        self._load_prev_stats_cache()

    @staticmethod
    def _payload_api_error(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
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

    @staticmethod
    def _is_rate_limited_error(text: str) -> bool:
        t = str(text or "").lower()
        return ("egw00201" in t) or ("초당 거래건수를 초과" in t) or ("rate limit" in t)

    def _record_api_diag(self, key: str, code: str, detail: str = "") -> None:
        self._api_diag_counts[key] = self._api_diag_counts.get(key, 0) + 1
        if detail:
            self._api_diag_last_by_code[str(code)] = str(detail)[:500]
            if len(self._api_diag_samples) < 60:
                self._api_diag_samples.append(
                    {
                        "key": str(key),
                        "code": str(code),
                        "detail": str(detail)[:500],
                    }
                )

    def _latest_api_diag_for(self, code: str) -> str:
        return str(self._api_diag_last_by_code.get(str(code), "") or "")

    def _log_api_call(
        self,
        *,
        kind: str,
        code: str,
        status: str,
        attempt: int,
        detail: str = "",
    ) -> None:
        msg = (
            f"[api-call] market=KR kind={kind} code={code} attempt={attempt} status={status}"
        )
        if detail:
            msg += f" detail={detail}"
        print(msg, flush=True)

    def get_api_diagnostics(self) -> Dict[str, Any]:
        return {
            "counts": dict(self._api_diag_counts),
            "sample_errors": list(self._api_diag_samples),
        }

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

    def _valid_universe_cache_path(self) -> Path:
        return self._cache_dir() / "kr_valid_universe.csv"

    def _benchmark_universe_cache_path(self) -> Path:
        return self._cache_dir() / "kr_benchmark_universe.csv"

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

    def _read_liquidity_cache(self, path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
        codes: List[str] = []
        names: Dict[str, str] = {}
        turnovers: Dict[str, float] = {}
        if not path.exists():
            return codes, names, turnovers
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
                    turnover = to_float(row.get("prev_turnover"))
                    if turnover is not None and turnover > 0:
                        turnovers[code] = turnover
        except Exception:
            return [], {}, {}
        return codes, names, turnovers

    def _write_liquidity_cache(self, path: Path, rows: List[Tuple[str, str, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["code", "name", "prev_turnover"])
            for code, name, turnover in rows:
                writer.writerow([code, name, f"{turnover:.0f}"])

    def _read_valid_universe_cache(self, path: Path) -> List[Tuple[str, str, PrevDayStats]]:
        rows: List[Tuple[str, str, PrevDayStats]] = []
        if not path.exists():
            return rows
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
                    if prev_turnover is None or prev_turnover <= 0:
                        continue
                    rows.append(
                        (
                            code,
                            str(row.get("name", "") or "").strip(),
                            PrevDayStats(
                                prev_close=float(prev_close),
                                prev_volume=float(prev_volume),
                                prev_turnover=float(prev_turnover),
                                prev_day_change_pct=prev_day_change_pct,
                            ),
                        )
                    )
        except Exception:
            return []
        return rows

    def _write_valid_universe_cache(self, path: Path, rows: List[Tuple[str, str, PrevDayStats]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["code", "name", "prev_close", "prev_volume", "prev_turnover", "prev_day_change_pct"]
            )
            for code, name, stats in rows:
                writer.writerow(
                    [
                        code,
                        name,
                        f"{stats.prev_close:.8f}",
                        f"{stats.prev_volume:.8f}",
                        f"{stats.prev_turnover:.8f}",
                        ""
                        if stats.prev_day_change_pct is None
                        else f"{stats.prev_day_change_pct:.8f}",
                    ]
                )

    def _read_benchmark_universe_cache(self, path: Path) -> Tuple[List[str], Dict[str, str], int, int]:
        codes: List[str] = []
        names: Dict[str, str] = {}
        seen = set()
        kospi_count = 0
        kosdaq_count = 0
        if not path.exists():
            return codes, names, kospi_count, kosdaq_count
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bucket = str(row.get("bucket", "")).strip().upper()
                    if bucket == "KOSPI200":
                        kospi_count += 1
                    elif bucket == "KOSDAQ150":
                        kosdaq_count += 1
                    code = normalize_code(row.get("code", "")).strip()
                    if len(code) != 6 or not code.isdigit() or code in seen:
                        continue
                    seen.add(code)
                    codes.append(code)
                    name = str(row.get("name", "")).strip()
                    if name:
                        names[code] = name
        except Exception:
            return [], {}, 0, 0
        return codes, names, kospi_count, kosdaq_count

    def _write_benchmark_universe_cache(self, path: Path, rows: List[Tuple[str, str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["code", "name", "bucket"])
            for code, name, bucket in rows:
                writer.writerow([code, name, bucket])

    @staticmethod
    def _flag_enabled(value: Any) -> bool:
        text = str(value if value is not None else "").strip().upper()
        if not text:
            return False
        if text in {"0", "N", "NO", "FALSE", "F", "-", "--", "NAN", "NONE"}:
            return False
        return True

    def _load_kr_benchmark_universe(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Build KR source universe from KOSPI200 + KOSDAQ150 constituents only.
        """
        cache_path = self._benchmark_universe_cache_path()
        cached_codes, cached_names, cached_kospi, cached_kosdaq = self._read_benchmark_universe_cache(
            cache_path
        )
        if cached_codes:
            print(
                f"[universe] KR source from daily cache: KOSPI200={cached_kospi}, "
                f"KOSDAQ150={cached_kosdaq}, merged={len(cached_codes)}, cache={cache_path}",
                flush=True,
            )
            return cached_codes, cached_names

        try:
            kospi_df = self.broker.fetch_kospi_symbols()
            kosdaq_df = self.broker.fetch_kosdaq_symbols()
        except Exception as exc:
            raise RuntimeError(f"KR benchmark universe fetch failed: {exc}") from exc

        if kospi_df is None or getattr(kospi_df, "empty", True):
            raise RuntimeError("KR benchmark universe fetch failed: empty_kospi_symbols")
        if kosdaq_df is None or getattr(kosdaq_df, "empty", True):
            raise RuntimeError("KR benchmark universe fetch failed: empty_kosdaq_symbols")

        kospi_flag_col = None
        for col in ("KOSPI200섹터업종", "KOSPI200"):
            if col in list(kospi_df.columns):
                kospi_flag_col = col
                break
        if not kospi_flag_col:
            cols = ",".join(str(c) for c in list(kospi_df.columns)[:20])
            raise RuntimeError(f"KR benchmark universe fetch failed: missing_kospi200_flag_col cols={cols}")
        if "KOSDAQ150" not in list(kosdaq_df.columns):
            cols = ",".join(str(c) for c in list(kosdaq_df.columns)[:20])
            raise RuntimeError(f"KR benchmark universe fetch failed: missing_kosdaq150_flag_col cols={cols}")

        codes: List[str] = []
        names: Dict[str, str] = {}
        seen = set()
        cache_rows: List[Tuple[str, str, str]] = []

        def _append(code_raw: Any, name_raw: Any, bucket: str) -> None:
            code = normalize_code(code_raw).strip()
            if len(code) != 6 or not code.isdigit():
                return
            if code in seen:
                return
            seen.add(code)
            codes.append(code)
            name = str(name_raw or "").strip()
            if name:
                names[code] = name
            cache_rows.append((code, name, bucket))

        kospi_count = 0
        for _, row in kospi_df.iterrows():
            if not self._flag_enabled(row.get(kospi_flag_col)):
                continue
            kospi_count += 1
            _append(row.get("단축코드"), row.get("한글명"), "KOSPI200")

        kosdaq_count = 0
        for _, row in kosdaq_df.iterrows():
            if not self._flag_enabled(row.get("KOSDAQ150")):
                continue
            kosdaq_count += 1
            _append(row.get("단축코드"), row.get("한글명"), "KOSDAQ150")

        if not codes:
            raise RuntimeError(
                "KR benchmark universe fetch failed: no_codes_from_kospi200_kosdaq150"
            )
        self._write_benchmark_universe_cache(cache_path, cache_rows)

        print(
            f"[universe] KR source fixed: KOSPI200={kospi_count}, KOSDAQ150={kosdaq_count}, "
            f"merged={len(codes)}, cache={cache_path}",
            flush=True,
        )
        return codes, names

    def _build_objective_universe(self) -> Tuple[List[str], Dict[str, str]]:
        liquidity_top_n = 20
        source_codes, source_names = self._load_kr_benchmark_universe()
        source_code_set = set(source_codes)
        cache_path = self._liquidity_cache_path()
        valid_cache_path = self._valid_universe_cache_path()
        valid_rows = self._read_valid_universe_cache(valid_cache_path)
        valid_by_code: Dict[str, Tuple[str, PrevDayStats]] = {
            code: (name, stats) for code, name, stats in valid_rows
        }
        cached_codes, cached_names, cached_turnovers = self._read_liquidity_cache(cache_path)
        if not cached_codes:
            legacy_path = self._legacy_liquidity_cache_path()
            if legacy_path.exists():
                cached_codes, cached_names, cached_turnovers = self._read_liquidity_cache(legacy_path)
        cached_codes = [code for code in cached_codes if code in source_code_set and code in valid_by_code]
        if cached_codes and len(cached_codes) >= liquidity_top_n:
            target = min(liquidity_top_n, len(cached_codes))
            selected_codes = cached_codes[:target]
            selected_names = {
                code: source_names.get(code, cached_names.get(code, "")) for code in selected_codes
            }
            for code in selected_codes:
                _, stats = valid_by_code[code]
                self._daily_cache[code] = stats
            print(
                f"[universe] KR objective pool from cache: {target}/{len(cached_codes)} "
                "(basis=valid_prev_day_stats+prev_day_turnover_rank, source=KOSPI200+KOSDAQ150)",
                flush=True,
            )
            return selected_codes, selected_names

        scan_codes = source_codes

        ranked: List[Tuple[str, str, float]] = []
        valid_for_cache: List[Tuple[str, str, PrevDayStats]] = []
        total = len(scan_codes)
        progress_every = 1
        print(
            f"[universe] KR validity scan start: total={total}, progress_every={progress_every}",
            flush=True,
        )
        for idx, code in enumerate(scan_codes, start=1):
            try:
                prev = self.fetch_prev_day_stats(code)
                if prev is None or prev.prev_close <= 0 or prev.prev_turnover <= 0:
                    continue
                name = source_names.get(code, "")
                ranked.append((code, name, prev.prev_turnover))
                valid_for_cache.append((code, name, prev))
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[universe] validity-scan={idx}/{total} ({pct:.1f}%), valid={len(valid_for_cache)}, ranked_buffer={len(ranked)}",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        if valid_for_cache:
            self._write_valid_universe_cache(valid_cache_path, valid_for_cache)
            print(
                f"[universe] KR valid universe cached: {len(valid_for_cache)} ({valid_cache_path})",
                flush=True,
            )
        print(f"[universe] KR liquidity rank sort start: valid={len(ranked)}", flush=True)
        ranked.sort(key=lambda item: item[2], reverse=True)
        print(f"[universe] KR liquidity rank sort done: ranked={len(ranked)}", flush=True)
        if ranked:
            self._write_liquidity_cache(cache_path, ranked)
            print(
                f"[universe] KR liquidity cache saved: {len(ranked)} ({cache_path})",
                flush=True,
            )

        target = min(liquidity_top_n, len(ranked))
        selected = ranked[:target]
        selected_codes = [code for code, _, _ in selected]
        selected_names = {code: name for code, name, _ in selected if name}
        print(
            f"[universe] KR objective pool built: {len(selected_codes)}/{len(ranked)} "
            f"(basis=valid_prev_day_stats+prev_day_turnover_rank, top_n={liquidity_top_n}, scanned={len(scan_codes)}, source=KOSPI200+KOSDAQ150)",
            flush=True,
        )
        return selected_codes, selected_names

    def load_universe(self) -> Tuple[List[str], Dict[str, str]]:
        if self.config.universe_file:
            universe_path = Path(self.config.universe_file)
            if not universe_path.exists():
                raise FileNotFoundError(f"Universe file not found: {universe_path}")
            codes, file_names = parse_universe_file(universe_path)
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
            raise RuntimeError("KR objective universe empty after KOSPI200+KOSDAQ150 scan")

        final_codes = objective_codes
        final_names = {code: objective_names.get(code, "") for code in final_codes}
        print(
            f"[universe] KR objective pool selected: {len(final_codes)} "
            "(basis=prev_day_turnover_rank_top20)",
            flush=True,
        )
        return final_codes, final_names

    def fetch_prev_day_stats(self, code: str) -> Optional[PrevDayStats]:
        if code in self._daily_cache:
            return self._daily_cache[code]

        rows: List[Dict[str, Any]] = []
        final_err = ""
        for attempt in range(self._rate_limit_retries + 1):
            resp: Dict[str, Any]
            self._log_api_call(
                kind="fetch_prev_day_stats",
                code=code,
                attempt=attempt + 1,
                status="start",
            )
            try:
                resp = self.broker.fetch_ohlcv_recent30(code, timeframe="D", adj_price=True)
            except Exception as exc:
                err_text = repr(exc)
                final_err = err_text
                self._record_api_diag("fetch_prev_day_exception", code, err_text)
                self._log_api_call(
                    kind="fetch_prev_day_stats",
                    code=code,
                    attempt=attempt + 1,
                    status="exception",
                    detail=err_text,
                )
                if self._is_rate_limited_error(err_text) and attempt < self._rate_limit_retries:
                    sleep_sec = min(
                        self._rate_limit_backoff_max_sec,
                        self._rate_limit_backoff_sec * (2**attempt),
                    )
                    self._log_api_call(
                        kind="fetch_prev_day_stats",
                        code=code,
                        attempt=attempt + 1,
                        status="retry_backoff",
                        detail=f"{sleep_sec:.2f}s",
                    )
                    time.sleep(sleep_sec)
                    continue
                self._daily_cache[code] = None
                return None

            err = self._payload_api_error(resp if isinstance(resp, dict) else {})
            if err:
                final_err = err
                self._record_api_diag("fetch_prev_day_api_error", code, err)
                if self._is_rate_limited_error(err):
                    self._record_api_diag("fetch_prev_day_rate_limited", code, err)
            rows = latest_list_of_dict(resp if isinstance(resp, dict) else {})
            self._log_api_call(
                kind="fetch_prev_day_stats",
                code=code,
                attempt=attempt + 1,
                status="response",
                detail=f"rows={len(rows)} err={err or '-'}",
            )
            if rows:
                break
            self._record_api_diag("fetch_prev_day_empty_rows", code, err or "empty_rows")
            if err and self._is_rate_limited_error(err) and attempt < self._rate_limit_retries:
                sleep_sec = min(
                    self._rate_limit_backoff_max_sec,
                    self._rate_limit_backoff_sec * (2**attempt),
                )
                self._log_api_call(
                    kind="fetch_prev_day_stats",
                    code=code,
                    attempt=attempt + 1,
                    status="retry_backoff",
                    detail=f"{sleep_sec:.2f}s",
                )
                time.sleep(sleep_sec)
                continue
            break

        if not rows and final_err:
            self._record_api_diag("fetch_prev_day_final_error", code, final_err)
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
            self._record_api_diag("fetch_prev_day_no_parsed_rows", code, "parsed_rows=0")
            self._daily_cache[code] = None
            return None

        parsed_rows.sort(key=lambda x: x[0], reverse=True)
        past_rows = [row for row in parsed_rows if row[0] < self.today_kst]
        target_rows = past_rows if past_rows else parsed_rows
        if not target_rows:
            self._record_api_diag("fetch_prev_day_no_target_rows", code, "target_rows=0")
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

        final_err = ""
        for attempt in range(self._rate_limit_retries + 1):
            self._log_api_call(
                kind="fetch_price_snapshot",
                code=code,
                attempt=attempt + 1,
                status="start",
            )
            try:
                resp = self.broker.fetch_price(code)
            except Exception as exc:
                err_text = repr(exc)
                final_err = err_text
                self._record_api_diag("fetch_price_exception", code, err_text)
                self._log_api_call(
                    kind="fetch_price_snapshot",
                    code=code,
                    attempt=attempt + 1,
                    status="exception",
                    detail=err_text,
                )
                if self._is_rate_limited_error(err_text) and attempt < self._rate_limit_retries:
                    sleep_sec = min(
                        self._rate_limit_backoff_max_sec,
                        self._rate_limit_backoff_sec * (2**attempt),
                    )
                    self._log_api_call(
                        kind="fetch_price_snapshot",
                        code=code,
                        attempt=attempt + 1,
                        status="retry_backoff",
                        detail=f"{sleep_sec:.2f}s",
                    )
                    time.sleep(sleep_sec)
                    continue
                self._price_cache[code] = None
                return None

            err = self._payload_api_error(resp if isinstance(resp, dict) else {})
            if err:
                final_err = err
                self._record_api_diag("fetch_price_api_error", code, err)
                if self._is_rate_limited_error(err):
                    self._record_api_diag("fetch_price_rate_limited", code, err)
            output = first_dict(resp if isinstance(resp, dict) else {})
            self._log_api_call(
                kind="fetch_price_snapshot",
                code=code,
                attempt=attempt + 1,
                status="response",
                detail=f"has_output={int(bool(output))} err={err or '-'}",
            )
            if not output:
                self._record_api_diag("fetch_price_empty_payload", code, err or "empty_payload")
                if err and self._is_rate_limited_error(err) and attempt < self._rate_limit_retries:
                    sleep_sec = min(
                        self._rate_limit_backoff_max_sec,
                        self._rate_limit_backoff_sec * (2**attempt),
                    )
                    self._log_api_call(
                        kind="fetch_price_snapshot",
                        code=code,
                        attempt=attempt + 1,
                        status="retry_backoff",
                        detail=f"{sleep_sec:.2f}s",
                    )
                    time.sleep(sleep_sec)
                    continue
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
                self._record_api_diag("fetch_price_no_price_field", code, err or "price_missing")
                if err and self._is_rate_limited_error(err) and attempt < self._rate_limit_retries:
                    sleep_sec = min(
                        self._rate_limit_backoff_max_sec,
                        self._rate_limit_backoff_sec * (2**attempt),
                    )
                    self._log_api_call(
                        kind="fetch_price_snapshot",
                        code=code,
                        attempt=attempt + 1,
                        status="retry_backoff",
                        detail=f"{sleep_sec:.2f}s",
                    )
                    time.sleep(sleep_sec)
                    continue
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
            self._log_api_call(
                kind="fetch_price_snapshot",
                code=code,
                attempt=attempt + 1,
                status="ok",
                detail=f"price={price}",
            )
            return snapshot

        if final_err:
            self._record_api_diag("fetch_price_final_error", code, final_err)
        self._price_cache[code] = None
        return None

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
        """
        Stage1 gate removed: objective pool (liquidity top-N) is promoted directly
        to final candidates with best-effort snapshot hydration.
        """
        candidates: List[Stage1Candidate] = []
        scan_rows: List[Dict[str, Any]] = []
        total = len(codes)
        progress_every = max(1, self.config.stage1_log_interval)
        limit = max(1, int(self.config.pre_candidates))

        for idx, code in enumerate(codes, start=1):
            scan_row: Dict[str, Any] = {
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
                "passed_stage1": True,
                "skip_reason": "",
                "min_change_pct": self.config.min_change_pct,
                "min_gap_pct": self.config.min_gap_pct,
                "min_prev_turnover": self.config.min_prev_turnover,
                "long_only": self.config.long_only,
                "mode": "stage1_removed_top_liquidity",
            }

            try:
                print(f"[candidate-api] code={code} step=prev_day_stats cache-check start", flush=True)
                prev = self._daily_cache.get(code)
                if prev is None or prev.prev_close <= 0 or prev.prev_turnover <= 0:
                    scan_row["passed_stage1"] = False
                    scan_row["skip_reason"] = "invalid_prev_day_stats_cache"
                    print(
                        f"[candidate-api] code={code} step=prev_day_stats fail reason=invalid_prev_day_stats_cache",
                        flush=True,
                    )
                    scan_rows.append(scan_row)
                    continue
                print(
                    f"[candidate-api] code={code} step=prev_day_stats cache-hit prev_close={prev.prev_close:.4f} prev_turnover={prev.prev_turnover:.0f}",
                    flush=True,
                )

                print(
                    f"[candidate-api] code={code} step=price_snapshot start",
                    flush=True,
                )
                snap = self.fetch_price_snapshot(code)
                prev_close = float(prev.prev_close)
                prev_volume = float(prev.prev_volume)
                prev_turnover = float(prev.prev_turnover)
                current_price = prev_close
                open_price = prev_close
                change_pct = 0.0
                if snap is not None:
                    snap_name = str(snap.get("name") or "").strip()
                    if snap_name and not scan_row.get("name"):
                        scan_row["name"] = snap_name
                    snap_price = to_float(snap.get("price"))
                    snap_open = to_float(snap.get("open"))
                    snap_change = to_float(snap.get("change_pct"))
                    if snap_price is not None and snap_price > 0:
                        current_price = snap_price
                    if snap_open is not None and snap_open > 0:
                        open_price = snap_open
                    else:
                        open_price = current_price
                    if snap_change is not None:
                        change_pct = snap_change
                    elif prev_close > 0:
                        change_pct = ((current_price - prev_close) / prev_close) * 100.0
                    print(
                        f"[candidate-api] code={code} step=price_snapshot ok price={current_price:.4f} open={open_price:.4f} change_pct={change_pct:.4f}",
                        flush=True,
                    )
                else:
                    scan_row["skip_reason"] = "no_price_snapshot_fallback_prev_close"
                    print(
                        f"[candidate-api] code={code} step=price_snapshot fail reason=no_price_snapshot_fallback_prev_close diag={self._latest_api_diag_for(code)}",
                        flush=True,
                    )
                    if prev_close > 0:
                        change_pct = ((current_price - prev_close) / prev_close) * 100.0

                if current_price <= 0:
                    current_price = prev_close
                if open_price <= 0:
                    open_price = prev_close
                gap_pct = ((open_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

                scan_row["current_price"] = current_price
                scan_row["open_price"] = open_price
                scan_row["change_pct"] = change_pct
                scan_row["gap_pct"] = gap_pct
                scan_row["prev_close"] = prev_close
                scan_row["prev_day_volume"] = prev_volume
                scan_row["prev_day_turnover"] = prev_turnover
                scan_row["pass_prev_turnover"] = True
                scan_rows.append(scan_row)

                candidates.append(
                    Stage1Candidate(
                        code=code,
                        name=str(scan_row.get("name", "") or ""),
                        current_price=current_price,
                        open_price=open_price,
                        current_change_pct=change_pct,
                        gap_pct=gap_pct,
                        prev_close=prev_close,
                        prev_day_volume=prev_volume,
                        prev_day_turnover=prev_turnover,
                    )
                )
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[candidate] scanned={idx}/{total} ({pct:.1f}%), selected={len(candidates)}",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        candidates.sort(key=lambda c: c.prev_day_turnover, reverse=True)
        self.last_stage1_scan = scan_rows
        selected = candidates[:limit]
        print(
            f"[candidate] selected={len(selected)}/{len(candidates)} "
            "(basis=liquidity_top_pool, stage1=disabled)",
            flush=True,
        )
        api_diag = self.get_api_diagnostics()
        if api_diag.get("counts"):
            print(f"[candidate-api-summary] counts={api_diag['counts']}", flush=True)
            for sample in list(api_diag.get("sample_errors", []))[:5]:
                print(
                    f"[candidate-api-sample] code={sample.get('code')} key={sample.get('key')} detail={sample.get('detail')}",
                    flush=True,
                )
        return selected

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
