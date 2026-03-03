from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

from systematic_alpha.models import Stage1Candidate
from systematic_alpha.trader.precompute import _market_filter, precompute_market
from systematic_alpha.trader.scheduler import should_run_budget_snapshot, should_run_precompute
from systematic_alpha.trader.storage import TraderStorage


KST = ZoneInfo("Asia/Seoul")


class _FakeKRBroker:
    def fetch_ohlcv_recent30(self, code: str, timeframe: str = "D", adj_price: bool = True):
        _ = (timeframe, adj_price)
        code = str(code).upper()
        if code == "069500":
            # index bars for market filter MA test (prev_close > MA3)
            return {
                "output1": [
                    {"stck_bsop_date": "20260227", "stck_hgpr": "101", "stck_lwpr": "99", "stck_clpr": "100"},
                    {"stck_bsop_date": "20260228", "stck_hgpr": "102", "stck_lwpr": "100", "stck_clpr": "101"},
                    {"stck_bsop_date": "20260302", "stck_hgpr": "103", "stck_lwpr": "101", "stck_clpr": "102"},
                ]
            }
        # candidate symbol prev-day range
        if code == "000001":
            return {"output1": [{"stck_bsop_date": "20260302", "stck_hgpr": "110", "stck_lwpr": "100", "stck_clpr": "105"}]}
        if code == "000002":
            return {"output1": [{"stck_bsop_date": "20260302", "stck_hgpr": "220", "stck_lwpr": "200", "stck_clpr": "210"}]}
        return {"output1": []}


class _FakeKRSelector:
    def __init__(self):
        self.broker = _FakeKRBroker()

    def load_universe(self):
        return ["000001", "000002"], {"000001": "AAA", "000002": "BBB"}

    def build_stage1_candidates(self, codes, names):
        _ = (codes, names)
        return [
            Stage1Candidate(
                code="000001",
                name="AAA",
                current_price=106.0,
                open_price=102.0,
                current_change_pct=1.0,
                gap_pct=0.5,
                prev_close=105.0,
                prev_day_volume=1000.0,
                prev_day_turnover=100000.0,
            ),
            Stage1Candidate(
                code="000002",
                name="BBB",
                current_price=215.0,
                open_price=210.0,
                current_change_pct=0.8,
                gap_pct=0.4,
                prev_close=210.0,
                prev_day_volume=2000.0,
                prev_day_turnover=200000.0,
            ),
        ]


class TraderMorningPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.storage = TraderStorage(self.root / "state" / "trader" / "trader.sqlite")

    def tearDown(self) -> None:
        self.storage.close()
        self._tmp.cleanup()

    def test_should_run_precompute_window(self) -> None:
        cfg = SimpleNamespace(precompute_start_kst="08:05", precompute_end_kst="08:33")
        self.assertFalse(should_run_precompute(cfg, datetime(2026, 3, 3, 8, 4, 59, tzinfo=KST)))
        self.assertTrue(should_run_precompute(cfg, datetime(2026, 3, 3, 8, 5, 0, tzinfo=KST)))
        self.assertTrue(should_run_precompute(cfg, datetime(2026, 3, 3, 8, 20, 0, tzinfo=KST)))
        self.assertFalse(should_run_precompute(cfg, datetime(2026, 3, 3, 8, 34, 0, tzinfo=KST)))

    def test_should_run_budget_snapshot_window(self) -> None:
        cfg = SimpleNamespace(budget_snapshot_kst="08:40")
        self.assertFalse(should_run_budget_snapshot(cfg, datetime(2026, 3, 3, 8, 39, 59, tzinfo=KST), grace_minutes=20))
        self.assertTrue(should_run_budget_snapshot(cfg, datetime(2026, 3, 3, 8, 40, 0, tzinfo=KST), grace_minutes=20))
        self.assertTrue(should_run_budget_snapshot(cfg, datetime(2026, 3, 3, 8, 59, 0, tzinfo=KST), grace_minutes=20))
        self.assertFalse(should_run_budget_snapshot(cfg, datetime(2026, 3, 3, 9, 1, 0, tzinfo=KST), grace_minutes=20))

    def test_precompute_market_persists_candidates_and_breakout(self) -> None:
        trade_date = "20260303"
        cfg = SimpleNamespace(
            out_dir=self.root / "out" / "trader",
            candidates_max_kr=20,
            candidates_max_us=20,
            k=0.5,
            market_filter_symbol_kr="069500",
            market_filter_symbol_us="SPY",
            market_filter_ma_days=3,
        )
        with mock.patch("systematic_alpha.trader.precompute.make_selector", return_value=_FakeKRSelector()):
            result = precompute_market(cfg=cfg, storage=self.storage, market="KR", trade_date=trade_date)

        self.assertEqual("OK", result["status"])
        self.assertEqual(2, result["candidate_count"])

        rows = self.storage.list_candidate_symbols(trade_date, "KR")
        self.assertEqual(2, len(rows))
        by_symbol = {r["symbol"]: r for r in rows}

        # breakout = today_open + (prev_high - prev_low) * k
        self.assertAlmostEqual(107.0, float(by_symbol["000001"]["breakout_price"]))
        self.assertAlmostEqual(220.0, float(by_symbol["000002"]["breakout_price"]))

        mf = self.storage.get_market_filter(trade_date, "KR")
        self.assertIsNotNone(mf)
        self.assertTrue(bool(mf["trading_enabled"]))

    def test_precompute_market_marks_error_on_universe_failure(self) -> None:
        trade_date = "20260303"
        cfg = SimpleNamespace(
            out_dir=self.root / "out" / "trader",
            candidates_max_kr=20,
            candidates_max_us=20,
            k=0.5,
            market_filter_symbol_kr="069500",
            market_filter_symbol_us="SPY",
            market_filter_ma_days=20,
        )

        broken_selector = SimpleNamespace(
            load_universe=mock.Mock(side_effect=RuntimeError("remote_source_unavailable"))
        )

        with mock.patch("systematic_alpha.trader.precompute.make_selector", return_value=broken_selector):
            result = precompute_market(cfg=cfg, storage=self.storage, market="US", trade_date=trade_date)

        self.assertEqual("ERROR", result["status"])
        self.assertEqual(0, result["candidate_count"])
        self.assertIn("remote_source_unavailable", str(result["detail"].get("error", "")))

    def test_market_filter_distinguishes_index_fetch_failed(self) -> None:
        with mock.patch(
            "systematic_alpha.trader.precompute._fetch_ohlcv_rows_with_diag",
            return_value=([], {"status": "fetch_failed", "error": "rt_cd=1,msg_cd=E123,msg1=fail"}),
        ):
            out = _market_filter(
                selector=object(),
                market="US",
                index_symbol="SPY",
                trade_date="20260303",
                ma_days=20,
            )
        self.assertEqual("index_fetch_failed", out["reason"])
        self.assertFalse(bool(out["trading_enabled"]))
        self.assertIn("fetch_diagnostics", out)

    def test_market_filter_distinguishes_insufficient_index_bars(self) -> None:
        rows = [
            {"date": "20260301", "close": 100.0},
            {"date": "20260302", "close": 101.0},
        ]
        with mock.patch(
            "systematic_alpha.trader.precompute._fetch_ohlcv_rows_with_diag",
            return_value=(rows, {"status": "ok", "bars": 2}),
        ):
            out = _market_filter(
                selector=object(),
                market="US",
                index_symbol="SPY",
                trade_date="20260303",
                ma_days=20,
            )
        self.assertEqual("insufficient_index_bars", out["reason"])
        self.assertEqual(2, int(out["bars"]))
        self.assertEqual(20, int(out["required_bars"]))


if __name__ == "__main__":
    unittest.main()
