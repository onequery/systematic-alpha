from __future__ import annotations

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

        symbols_df = self.broker.fetch_symbols()
        return extract_codes_and_names_from_df(symbols_df, self.config.max_symbols_scan)

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

    def fetch_price_snapshot(self, code: str) -> Optional[Dict[str, Any]]:
        if code in self._price_cache:
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

    def _build_candidates_with_thresholds(
        self,
        codes: List[str],
        names: Dict[str, str],
        min_change_pct: float,
        min_gap_pct: float,
        min_prev_turnover: float,
        limit: int,
    ) -> List[Stage1Candidate]:
        candidates: List[Stage1Candidate] = []
        total = len(codes)
        progress_every = max(1, self.config.stage1_log_interval)
        for idx, code in enumerate(codes, start=1):
            try:
                snap = self.fetch_price_snapshot(code)
                if snap is None:
                    continue

                current_price = snap["price"]
                open_price = snap["open"]
                change_pct = snap["change_pct"]
                if current_price is None or open_price is None:
                    continue

                if change_pct is not None and abs(change_pct) < min_change_pct:
                    continue

                prev = self.fetch_prev_day_stats(code)
                if prev is None or prev.prev_close <= 0:
                    continue

                if change_pct is None:
                    change_pct = ((current_price - prev.prev_close) / prev.prev_close) * 100.0

                if abs(change_pct) < min_change_pct:
                    continue

                gap_pct = ((open_price - prev.prev_close) / prev.prev_close) * 100.0
                if abs(gap_pct) < min_gap_pct:
                    continue

                if prev.prev_turnover < min_prev_turnover:
                    continue

                candidates.append(
                    Stage1Candidate(
                        code=code,
                        name=names.get(code, "") or str(snap.get("name") or ""),
                        current_price=current_price,
                        open_price=open_price,
                        current_change_pct=change_pct,
                        gap_pct=gap_pct,
                        prev_close=prev.prev_close,
                        prev_day_volume=prev.prev_volume,
                        prev_day_turnover=prev.prev_turnover,
                    )
                )
            finally:
                if idx % progress_every == 0 or idx == total:
                    pct = (idx / total * 100.0) if total > 0 else 100.0
                    print(
                        f"[stage1] scanned={idx}/{total} ({pct:.1f}%), candidates={len(candidates)}",
                        flush=True,
                    )
                if self.config.rest_sleep_sec > 0:
                    time.sleep(self.config.rest_sleep_sec)

        candidates.sort(
            key=lambda c: (c.prev_day_turnover, abs(c.current_change_pct), abs(c.gap_pct)),
            reverse=True,
        )
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

        total_ask = to_float(values[43])
        total_bid = to_float(values[44])
        if total_ask is None or total_bid is None or total_ask <= 0:
            return
        ref.bid_ask_ratios.append(total_bid / total_ask)

    def collect_realtime(self, codes: List[str]) -> Tuple[Dict[str, RealtimeStats], bool]:
        stats = {code: RealtimeStats() for code in codes}
        if not codes or self.config.collect_seconds <= 0:
            return stats, False

        log_interval = max(1, self.config.realtime_log_interval)
        ws = self.mojito.KoreaInvestmentWS(
            self.config.api_key,
            self.config.api_secret,
            ["H0STCNT0", "H0STASP0"],
            codes,
            user_id=self.config.user_id,
        )

        print(
            f"[realtime] starting websocket: codes={len(codes)}, duration={self.config.collect_seconds}s, heartbeat={log_interval}s",
            flush=True,
        )
        ws.start()
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
        print(
            "[realtime] finished: "
            f"exec_events={execution_events}, orderbook_events={orderbook_events}, "
            f"exec_symbols={exec_symbols}/{len(codes)}, orderbook_symbols={ob_symbols}/{len(codes)}",
            flush=True,
        )

        return stats, execution_events > 0

    def evaluate(
        self, candidates: List[Stage1Candidate], stats: Dict[str, RealtimeStats], realtime_ready: bool
    ) -> List[FinalSelection]:
        results: List[FinalSelection] = []

        for candidate in candidates:
            realtime = stats.get(candidate.code, RealtimeStats())
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
                    "change_pct": abs(candidate.current_change_pct) >= self.config.min_change_pct,
                    "gap_pct": abs(candidate.gap_pct) >= self.config.min_gap_pct,
                    "prev_turnover": candidate.prev_day_turnover >= self.config.min_prev_turnover,
                    "strength_maintained": strength_ok,
                    "volume_ratio": volume_ratio is not None
                    and volume_ratio >= self.config.min_vol_ratio,
                    "bid_ask_maintained": bid_ask_ok,
                    "price_above_vwap": current_vs_vwap,
                    "low_not_broken": not realtime.low_broken_after_start,
                }
                max_score = 8
                pass_cut = self.config.min_pass_conditions
            else:
                conditions = {
                    "change_pct": abs(candidate.current_change_pct) >= self.config.min_change_pct,
                    "gap_pct": abs(candidate.gap_pct) >= self.config.min_gap_pct,
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
