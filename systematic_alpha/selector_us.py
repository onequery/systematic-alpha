from __future__ import annotations

import csv
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import requests

from systematic_alpha.helpers import (
    first_dict,
    latest_list_of_dict,
    maintained,
    normalize_symbol,
    normalize_yyyymmdd,
    parse_us_universe_file,
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

SP500_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
SP500_BUNDLED_SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "us_sp500_snapshot.csv"


def _normalize_us_exchange_code(raw_exchange: str) -> str:
    raw = str(raw_exchange or "").strip()
    upper = raw.upper()
    return {
        "NASD": "NASD",
        "NASDAQ": "NASD",
        "NYSE": "NYSE",
        "NYS": "NYSE",
        "AMEX": "AMEX",
        "AMS": "AMEX",
    }.get(upper, upper or "NASD")


def _resolve_us_exchange_for_mojito(mojito_module, raw_exchange: str) -> str:
    # self-heal:us-exchange-resolver-v1
    raw = str(raw_exchange or "").strip()
    upper = raw.upper()
    normalized = {
        "NASD": "NASD",
        "NASDAQ": "NASD",
        "NYSE": "NYSE",
        "NYS": "NYSE",
        "AMEX": "AMEX",
        "AMS": "AMEX",
    }.get(upper, upper)

    try:
        ki = importlib.import_module(mojito_module.__name__ + ".koreainvestment")
        exchange_code3 = getattr(ki, "EXCHANGE_CODE3", {})
        if isinstance(exchange_code3, dict):
            for label, code in exchange_code3.items():
                if str(code).upper() == normalized:
                    return str(label)
    except Exception:
        pass

    fallback = {
        "NASD": "나스닥",
        "NYSE": "뉴욕",
        "AMEX": "아멕스",
    }
    return fallback.get(normalized, raw or "나스닥")


class USDayTradingSelector:
    def __init__(self, mojito_module, config: StrategyConfig):
        self.mojito = mojito_module
        self.config = config
        self._preferred_exchange_code = _normalize_us_exchange_code(config.us_exchange or "")
        self._broker_exchange = _resolve_us_exchange_for_mojito(self.mojito, self._preferred_exchange_code)
        self._broker_by_exchange: Dict[str, Any] = {}
        self._symbol_exchange_hint: Dict[str, str] = {}
        self._api_diag_counts: Dict[str, int] = {}
        self._api_diag_samples: List[Dict[str, str]] = []
        self.broker = self._get_broker_for_exchange(self._preferred_exchange_code)
        self.today_us = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        self.today_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        self._daily_cache: Dict[str, Optional[PrevDayStats]] = {}
        self._price_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._detail_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._daily_bars_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._orderbook_available = False
        self.last_stage1_scan: List[Dict[str, Any]] = []
        self._load_prev_stats_cache()

    def _get_broker_for_exchange(self, exchange_code: str):
        code = _normalize_us_exchange_code(exchange_code)
        if code in self._broker_by_exchange:
            return self._broker_by_exchange[code]
        try:
            broker_exchange = _resolve_us_exchange_for_mojito(self.mojito, code)
            broker = self.mojito.KoreaInvestment(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                acc_no=self.config.acc_no,
                exchange=broker_exchange,
                mock=self.config.mock,
            )
            self._broker_by_exchange[code] = broker
            return broker
        except Exception as exc:
            self._record_api_diag("broker_create_exception", code, repr(exc))
            return None

    def _exchange_attempt_order(self, code: str, preferred_exchange: Optional[str] = None) -> List[str]:
        symbol = normalize_symbol(code)
        ordered: List[str] = []
        if preferred_exchange:
            ordered.append(_normalize_us_exchange_code(preferred_exchange))
        hinted = self._symbol_exchange_hint.get(symbol)
        if hinted:
            ordered.append(_normalize_us_exchange_code(hinted))
        ordered.append(self._preferred_exchange_code)
        ordered.extend(["NASD", "NYSE", "AMEX"])
        out: List[str] = []
        seen = set()
        for ex in ordered:
            ex_norm = _normalize_us_exchange_code(ex)
            if not ex_norm or ex_norm in seen:
                continue
            seen.add(ex_norm)
            out.append(ex_norm)
        return out

    def _remember_symbol_exchange(self, code: str, exchange_code: str) -> None:
        symbol = normalize_symbol(code)
        if not symbol:
            return
        self._symbol_exchange_hint[symbol] = _normalize_us_exchange_code(exchange_code)

    def _record_api_diag(self, key: str, code: str, detail: str = "") -> None:
        self._api_diag_counts[key] = self._api_diag_counts.get(key, 0) + 1
        if detail and len(self._api_diag_samples) < 60:
            self._api_diag_samples.append(
                {
                    "key": key,
                    "code": code,
                    "detail": detail[:500],
                }
            )

    def get_api_diagnostics(self) -> Dict[str, Any]:
        return {
            "exchange_input": self.config.us_exchange,
            "exchange_broker": self._broker_exchange,
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
        return Path("out") / "us" / self.today_kst

    def _cache_dir(self) -> Path:
        return self._session_root_dir() / "cache"

    def _legacy_sp500_cache_path(self) -> Path:
        return Path("out") / f"us_sp500_constituents_{self.today_us}.csv"

    def _legacy_prev_stats_cache_path(self) -> Path:
        return Path("out") / f"us_prev_day_stats_{self.today_us}.csv"

    def _sp500_cache_path(self) -> Path:
        return self._cache_dir() / "us_sp500_constituents.csv"

    def _prev_stats_cache_path(self) -> Path:
        return self._cache_dir() / "us_prev_day_stats.csv"

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
                        symbol = normalize_symbol(row.get("code", ""))
                        if not symbol:
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
                        self._daily_cache[symbol] = PrevDayStats(
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
            symbol = normalize_symbol(code)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            unique_codes.append(symbol)

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

    @staticmethod
    def _read_sp500_csv(path: Path) -> Tuple[List[str], Dict[str, str]]:
        symbols: List[str] = []
        names: Dict[str, str] = {}
        seen = set()
        if not path.exists():
            return symbols, names

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
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
        except Exception:
            return [], {}
        return symbols, names

    @staticmethod
    def _write_sp500_csv(path: Path, symbols: List[str], names: Dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "name"])
            for symbol in symbols:
                writer.writerow([symbol, names.get(symbol, "")])

    def _fetch_sp500_remote(self) -> Tuple[List[str], Dict[str, str]]:
        try:
            resp = requests.get(SP500_SOURCE_URL, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[universe] US objective remote fetch failed: {exc}", flush=True)
            return [], {}

        try:
            reader = csv.DictReader(resp.text.splitlines())
        except Exception:
            return [], {}

        symbols: List[str] = []
        names: Dict[str, str] = {}
        seen = set()
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

    def _build_objective_universe(self) -> Tuple[List[str], Dict[str, str]]:
        source = ""
        symbols: List[str] = []
        names: Dict[str, str] = {}

        cache_path = self._sp500_cache_path()
        cached_symbols, cached_names = self._read_sp500_csv(cache_path)
        if not cached_symbols:
            legacy_path = self._legacy_sp500_cache_path()
            if legacy_path.exists():
                cached_symbols, cached_names = self._read_sp500_csv(legacy_path)
        if cached_symbols:
            source = f"cache:{cache_path}"
            symbols = cached_symbols
            names = cached_names
        else:
            remote_symbols, remote_names = self._fetch_sp500_remote()
            if remote_symbols:
                source = f"remote:{SP500_SOURCE_URL}"
                symbols = remote_symbols
                names = remote_names
                try:
                    self._write_sp500_csv(cache_path, symbols, names)
                except Exception:
                    pass
            else:
                bundled_symbols, bundled_names = self._read_sp500_csv(SP500_BUNDLED_SNAPSHOT_PATH)
                if bundled_symbols:
                    source = f"bundled:{SP500_BUNDLED_SNAPSHOT_PATH}"
                    symbols = bundled_symbols
                    names = bundled_names

        if not symbols:
            print(
                "[universe] US objective pool unavailable (S&P500 source and snapshot failed).",
                flush=True,
            )
            return [], {}

        target = min(self.config.us_universe_size, len(symbols))
        selected_symbols = symbols[:target]
        selected_names = {symbol: names.get(symbol, "") for symbol in selected_symbols if names.get(symbol)}
        print(
            f"[universe] US objective pool selected: {len(selected_symbols)}/{len(symbols)} "
            f"(basis=S&P500_constituents, source={source}, us_universe_size={self.config.us_universe_size})",
            flush=True,
        )
        return selected_symbols, selected_names

    def load_universe(self) -> Tuple[List[str], Dict[str, str]]:
        symbols: List[str]
        names: Dict[str, str]

        if self.config.universe_file:
            universe_path = Path(self.config.universe_file)
            if not universe_path.exists():
                raise FileNotFoundError(f"US universe file not found: {universe_path}")
            symbols, names = parse_us_universe_file(universe_path)
        else:
            symbols, names = self._build_objective_universe()

        deduped: List[str] = []
        deduped_names: Dict[str, str] = {}
        seen = set()
        for symbol in symbols:
            normalized = normalize_symbol(symbol)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
            if names.get(symbol):
                deduped_names[normalized] = names[symbol]
            elif names.get(normalized):
                deduped_names[normalized] = names[normalized]
            if len(deduped) >= self.config.max_symbols_scan:
                break

        return deduped, deduped_names

    @staticmethod
    def _parse_price_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        price = to_float(
            pick_first(
                payload,
                (
                    "last",
                    "last_price",
                    "trade_price",
                    "ovrs_nmix_prpr",
                    "ovrs_prpr",
                    "stck_prpr",
                    "close",
                    "clos",
                ),
            )
        )
        open_price = to_float(
            pick_first(
                payload,
                (
                    "open",
                    "open_price",
                    "ovrs_oprc",
                    "stck_oprc",
                    "oprc",
                ),
            )
        )
        change_pct = to_float(
            pick_first(
                payload,
                (
                    "prdy_ctrt",
                    "change_rate",
                    "chg_rt",
                    "rate",
                    "ovrs_prdy_ctrt",
                    "prdy_vrss_rt",
                ),
            )
        )
        acml_volume = to_float(
            pick_first(
                payload,
                (
                    "acml_vol",
                    "volume",
                    "tvol",
                    "ovrs_vol",
                    "cum_volume",
                ),
            )
        )
        low_price = to_float(
            pick_first(
                payload,
                (
                    "stck_lwpr",
                    "low",
                    "ovrs_lwpr",
                    "lwpr",
                    "day_low",
                ),
            )
        )
        stock_name_raw = pick_first(
            payload,
            (
                "prdt_name",
                "ovrs_item_name",
                "hts_kor_isnm",
                "name",
                "symbol_name",
            ),
        )
        stock_name = str(stock_name_raw).strip() if stock_name_raw else ""
        strength = to_float(
            pick_first(
                payload,
                (
                    "strength",
                    "cntg_strength",
                    "tday_rltv",
                    "exec_strength",
                ),
            )
        )
        total_ask = to_float(
            pick_first(
                payload,
                (
                    "total_ask_qty",
                    "ask_rsqn",
                    "ask_qty",
                    "tot_ask_qty",
                    "aspr_rsqn",
                ),
            )
        )
        total_bid = to_float(
            pick_first(
                payload,
                (
                    "total_bid_qty",
                    "bid_rsqn",
                    "bid_qty",
                    "tot_bid_qty",
                    "bidp_rsqn",
                ),
            )
        )
        vwap = to_float(
            pick_first(
                payload,
                (
                    "vwap",
                    "vwap_price",
                    "wght_avg_pric",
                    "avg_price",
                    "avrg_pric",
                ),
            )
        )
        return {
            "price": price,
            "open": open_price,
            "change_pct": change_pct,
            "acml_volume": acml_volume,
            "low_price": low_price,
            "name": stock_name,
            "strength": strength,
            "total_ask": total_ask,
            "total_bid": total_bid,
            "vwap": vwap,
        }

    def fetch_price_snapshot(self, code: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        if use_cache and code in self._price_cache:
            return self._price_cache[code]

        for exchange_code in self._exchange_attempt_order(code):
            broker = self._get_broker_for_exchange(exchange_code)
            if broker is None:
                continue
            try:
                resp = broker.fetch_price(code)
            except Exception as exc:
                self._record_api_diag("fetch_price_exception", code, f"{exchange_code}:{repr(exc)}")
                continue

            payload = first_dict(resp if isinstance(resp, dict) else {})
            if not payload and isinstance(resp, dict):
                payload = resp
            if not payload:
                detail = ""
                if isinstance(resp, dict):
                    detail = f"{exchange_code}:rt_cd={resp.get('rt_cd')} msg_cd={resp.get('msg_cd')} msg1={resp.get('msg1')}"
                self._record_api_diag("fetch_price_empty_payload", code, detail)
                continue

            parsed = self._parse_price_payload(payload)
            if parsed.get("price") is None:
                detail = ""
                if isinstance(resp, dict):
                    detail = f"{exchange_code}:rt_cd={resp.get('rt_cd')} msg_cd={resp.get('msg_cd')} msg1={resp.get('msg1')}"
                self._record_api_diag("fetch_price_no_price_field", code, detail)
                continue

            if parsed.get("open") is None or parsed.get("acml_volume") is None or parsed.get("low_price") is None:
                detail_row = self.fetch_price_detail_snapshot(
                    code,
                    use_cache=use_cache,
                    preferred_exchange=exchange_code,
                )
                if detail_row:
                    if parsed.get("open") is None:
                        parsed["open"] = detail_row.get("open")
                    if parsed.get("acml_volume") is None:
                        parsed["acml_volume"] = detail_row.get("acml_volume")
                    if parsed.get("low_price") is None:
                        parsed["low_price"] = detail_row.get("low_price")
                    if not parsed.get("name") and detail_row.get("name"):
                        parsed["name"] = detail_row.get("name")

            self._remember_symbol_exchange(code, exchange_code)
            self._price_cache[code] = parsed
            return parsed

        self._price_cache[code] = None
        return None

    def fetch_price_detail_snapshot(
        self,
        code: str,
        use_cache: bool = True,
        preferred_exchange: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if use_cache and code in self._detail_cache:
            return self._detail_cache[code]

        for exchange_code in self._exchange_attempt_order(code, preferred_exchange=preferred_exchange):
            broker = self._get_broker_for_exchange(exchange_code)
            if broker is None:
                continue
            try:
                resp = broker.fetch_price_detail_oversea(code)
            except Exception as exc:
                self._record_api_diag("fetch_price_detail_exception", code, f"{exchange_code}:{repr(exc)}")
                continue

            payload = first_dict(resp if isinstance(resp, dict) else {})
            if not payload and isinstance(resp, dict):
                payload = resp
            if not payload:
                detail = ""
                if isinstance(resp, dict):
                    detail = f"{exchange_code}:rt_cd={resp.get('rt_cd')} msg_cd={resp.get('msg_cd')} msg1={resp.get('msg1')}"
                self._record_api_diag("fetch_price_detail_empty_payload", code, detail)
                continue

            parsed = self._parse_price_payload(payload)
            self._remember_symbol_exchange(code, exchange_code)
            self._detail_cache[code] = parsed
            return parsed

        self._detail_cache[code] = None
        return None

    def fetch_prev_day_stats(self, code: str) -> Optional[PrevDayStats]:
        if code in self._daily_cache:
            return self._daily_cache[code]

        rows: List[Dict[str, Any]] = []
        resp: Dict[str, Any] = {}
        used_exchange = ""
        for exchange_code in self._exchange_attempt_order(code):
            broker = self._get_broker_for_exchange(exchange_code)
            if broker is None:
                continue
            try:
                candidate_resp = broker.fetch_ohlcv_oversea(code, timeframe="D", adj_price=True)
            except Exception as exc:
                self._record_api_diag("fetch_prev_day_exception", code, f"{exchange_code}:{repr(exc)}")
                continue
            resp = candidate_resp if isinstance(candidate_resp, dict) else {}
            rows = latest_list_of_dict(resp)
            if rows:
                used_exchange = exchange_code
                break
            detail = f"{exchange_code}:rt_cd={resp.get('rt_cd')} msg_cd={resp.get('msg_cd')} msg1={resp.get('msg1')}"
            self._record_api_diag("fetch_prev_day_empty_rows", code, detail)

        if not rows:
            self._daily_cache[code] = None
            return None

        if used_exchange:
            self._remember_symbol_exchange(code, used_exchange)

        parsed_rows: List[Tuple[str, float, float, float, Optional[float]]] = []
        for row in rows:
            date_key = pick_first(
                row,
                (
                    "xymd",
                    "date",
                    "stck_bsop_date",
                    "bas_dt",
                    "trdt_ymd",
                    "biz_dt",
                ),
            )
            date = normalize_yyyymmdd(date_key)
            if not date:
                continue

            close = to_float(
                pick_first(
                    row,
                    (
                        "clos",
                        "close",
                        "last",
                        "ovrs_nmix_prpr",
                        "stck_clpr",
                    ),
                )
            )
            volume = to_float(
                pick_first(
                    row,
                    (
                        "tvol",
                        "volume",
                        "acml_vol",
                        "ovrs_vol",
                    ),
                )
            )
            turnover = to_float(
                pick_first(
                    row,
                    (
                        "tamt",
                        "turnover",
                        "trade_value",
                        "acml_tr_pbmn",
                    ),
                )
            )
            day_change_pct = to_float(
                pick_first(
                    row,
                    (
                        "rate",
                        "prdy_ctrt",
                        "change_rate",
                        "ovrs_prdy_ctrt",
                    ),
                )
            )

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
        past_rows = [row for row in parsed_rows if row[0] < self.today_us]
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
        orderbook_ok = True
        if self._orderbook_available:
            orderbook_ok = realtime.orderbook_ticks >= self.config.min_orderbook_ticks
        return (
            realtime.execution_ticks >= self.config.min_exec_ticks
            and orderbook_ok
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

        rows: List[Dict[str, Any]] = []
        used_exchange = ""
        for exchange_code in self._exchange_attempt_order(code):
            broker = self._get_broker_for_exchange(exchange_code)
            if broker is None:
                continue
            try:
                resp = broker.fetch_ohlcv_oversea(code, timeframe="D", adj_price=True)
            except Exception as exc:
                self._record_api_diag("fetch_daily_bars_exception", code, f"{exchange_code}:{repr(exc)}")
                continue
            rows = latest_list_of_dict(resp if isinstance(resp, dict) else {})
            if rows:
                used_exchange = exchange_code
                break
        if not rows:
            self._daily_bars_cache[code] = []
            return []
        if used_exchange:
            self._remember_symbol_exchange(code, used_exchange)

        parsed: List[Dict[str, Any]] = []
        for row in rows:
            date = normalize_yyyymmdd(
                pick_first(row, ("xymd", "date", "stck_bsop_date", "bas_dt", "trdt_ymd", "biz_dt"))
            )
            if not date:
                continue
            open_price = to_float(pick_first(row, ("open", "oprc", "ovrs_oprc", "stck_oprc")))
            close_price = to_float(
                pick_first(row, ("clos", "close", "last", "ovrs_nmix_prpr", "stck_clpr"))
            )
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

        if entry_price is not None and entry_price > 0 and selection_close is not None:
            intraday_return_pct = ((selection_close - entry_price) / entry_price) * 100.0

        if selection_close is not None and selection_close > 0 and next_open is not None:
            overnight_return_pct = ((next_open - selection_close) / selection_close) * 100.0

        if entry_price is not None and entry_price > 0 and next_open is not None:
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
        for idx, symbol in enumerate(codes, start=1):
            scan_row: Optional[Dict[str, Any]]
            if record_scan:
                scan_row = {
                    "scan_index": idx,
                    "code": symbol,
                    "name": names.get(symbol, ""),
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
                snap = self.fetch_price_snapshot(symbol)
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

                prev = self.fetch_prev_day_stats(symbol)
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

                candidate_name = names.get(symbol, "") or str(snap.get("name") or "")
                candidates.append(
                    Stage1Candidate(
                        code=symbol,
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

        poll_interval = max(0.2, self.config.us_poll_interval)
        log_interval = max(1, self.config.realtime_log_interval)

        print(
            f"[realtime] starting US polling: codes={len(codes)}, duration={self.config.collect_seconds}s, "
            f"poll={poll_interval:.1f}s, heartbeat={log_interval}s",
            flush=True,
        )

        execution_events = 0
        orderbook_events = 0
        first_exec_logged = False
        first_orderbook_logged = False
        started = time.time()
        deadline = started + self.config.collect_seconds
        next_log = started + log_interval
        prev_acml_volume: Dict[str, Optional[float]] = {code: None for code in codes}
        prev_price: Dict[str, Optional[float]] = {code: None for code in codes}
        up_ticks: Dict[str, int] = {code: 0 for code in codes}
        down_ticks: Dict[str, int] = {code: 0 for code in codes}

        while time.time() < deadline:
            now = time.time()
            if now >= next_log:
                remain = max(int(deadline - now), 0)
                exec_symbols = sum(1 for ref in stats.values() if ref.got_execution)
                ob_symbols = sum(1 for ref in stats.values() if ref.got_orderbook)
                print(
                    "[realtime] "
                    f"remain={remain}s, "
                    f"exec_events={execution_events}, orderbook_events={orderbook_events}, "
                    f"exec_symbols={exec_symbols}/{len(codes)}, orderbook_symbols={ob_symbols}/{len(codes)}",
                    flush=True,
                )
                next_log += log_interval

            cycle_started = time.time()
            for code in codes:
                ref = stats[code]
                snap = self.fetch_price_snapshot(code, use_cache=False)
                detail = self.fetch_price_detail_snapshot(code, use_cache=False)

                if snap is None and detail is None:
                    continue

                latest_price = None
                if snap and snap.get("price") is not None:
                    latest_price = snap["price"]
                elif detail and detail.get("price") is not None:
                    latest_price = detail["price"]

                if latest_price is not None:
                    ref.latest_price = latest_price

                acml_volume: Optional[float] = None
                if snap:
                    acml_volume = snap.get("acml_volume")
                if acml_volume is None and detail:
                    acml_volume = detail.get("acml_volume")

                if acml_volume is not None:
                    ref.latest_acml_volume = acml_volume

                day_low = None
                if snap:
                    day_low = snap.get("low_price")
                if day_low is None and detail:
                    day_low = detail.get("low_price")
                if day_low is not None:
                    if ref.first_reported_low is None:
                        ref.first_reported_low = day_low
                    elif day_low < ref.first_reported_low:
                        ref.low_broken_after_start = True

                delta_volume: Optional[float] = None
                prev_vol = prev_acml_volume.get(code)
                if acml_volume is not None:
                    if prev_vol is not None and acml_volume > prev_vol:
                        delta_volume = acml_volume - prev_vol
                    prev_acml_volume[code] = acml_volume
                elif latest_price is not None:
                    delta_volume = 1.0

                if delta_volume is not None and delta_volume > 0:
                    ref.cum_trade_volume += delta_volume
                    if latest_price is not None:
                        ref.cum_trade_value += latest_price * delta_volume

                strength = None
                if detail:
                    strength = detail.get("strength")
                if strength is not None and strength > 0:
                    ref.strength_values.append(strength)
                elif latest_price is not None:
                    prev_px = prev_price.get(code)
                    if prev_px is not None:
                        if latest_price > prev_px:
                            up_ticks[code] += 1
                        elif latest_price < prev_px:
                            down_ticks[code] += 1
                        if down_ticks[code] > 0:
                            ref.strength_values.append((up_ticks[code] / down_ticks[code]) * 100.0)
                        elif up_ticks[code] > 0:
                            ref.strength_values.append(200.0)
                    prev_price[code] = latest_price

                bid_qty = detail.get("total_bid") if detail else None
                ask_qty = detail.get("total_ask") if detail else None
                if bid_qty is not None and ask_qty is not None and ask_qty > 0:
                    ref.got_orderbook = True
                    ref.orderbook_ticks += 1
                    ref.bid_ask_ratios.append(bid_qty / ask_qty)
                    orderbook_events += 1
                    if not first_orderbook_logged:
                        print("[realtime] first orderbook tick received.", flush=True)
                        first_orderbook_logged = True

                ref.got_execution = True
                ref.execution_ticks += 1
                execution_events += 1
                if not first_exec_logged:
                    print("[realtime] first execution tick received.", flush=True)
                    first_exec_logged = True

            elapsed_cycle = time.time() - cycle_started
            if elapsed_cycle < poll_interval:
                time.sleep(poll_interval - elapsed_cycle)

        self._orderbook_available = orderbook_events > 0

        exec_symbols = sum(1 for ref in stats.values() if ref.got_execution)
        ob_symbols = sum(1 for ref in stats.values() if ref.got_orderbook)
        eligible_count = sum(1 for ref in stats.values() if self._is_realtime_symbol_eligible(ref))
        total_count = len(codes)
        coverage_ratio = (eligible_count / total_count) if total_count > 0 else 0.0
        realtime_ready = execution_events > 0
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
                bid_ask_condition = (
                    symbol_realtime_eligible and bid_ask_ok
                    if self._orderbook_available
                    else True
                )
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
                    "bid_ask_maintained": bid_ask_condition,
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

