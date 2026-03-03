from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import load_trader_config
from systematic_alpha.trader.cli import _ensure_daily_bootstrap, _snapshot_budget
from systematic_alpha.trader.execution import execute_entry_intents
from systematic_alpha.trader.realtime import SignalIntent, _watch_symbols, collect_breakout_intents
from systematic_alpha.trader.scheduler import ET, us_liquidation_plan
from systematic_alpha.trader.storage import TraderStorage
from systematic_alpha.trader.sync import AccountSyncService, SyncResult, _api_error_text


class _DummyTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> bool:
        self.messages.append(str(text))
        return True


class _FakeSyncService:
    def __init__(self, results: list[SyncResult]):
        self._results = list(results)
        self.calls = 0

    def sync_account(self, *_args, **_kwargs) -> SyncResult:
        idx = min(self.calls, max(0, len(self._results) - 1))
        self.calls += 1
        return self._results[idx]


class TraderAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.storage = TraderStorage(self.root / "state" / "trader" / "trader.sqlite")

    def tearDown(self) -> None:
        self.storage.close()
        self._tmp.cleanup()

    def test_watch_symbols_is_candidates_union_server_holdings(self) -> None:
        trade_date = "20260303"
        self.storage.upsert_symbol_plan_rows(
            [
                {
                    "trade_date": trade_date,
                    "market": "KR",
                    "symbol": "AAA",
                    "name": "AAA",
                    "candidate_rank": 1,
                    "prev_high": 110.0,
                    "prev_low": 100.0,
                    "today_open": 101.0,
                    "breakout_price": 106.0,
                    "last_price": 105.0,
                    "is_candidate": True,
                }
            ]
        )
        self.storage.insert_account_snapshot(
            trade_date=trade_date,
            market_scope="KR",
            source="unit",
            strict=True,
            ok=True,
            blocked=False,
            reason="",
            cash_krw=1000000.0,
            equity_krw=1000000.0,
            payload={"ok": True},
            positions=[
                {
                    "market": "KR",
                    "symbol": "BBB",
                    "quantity": 7.0,
                    "avg_price": 5000.0,
                    "market_value_krw": 35000.0,
                    "currency": "KRW",
                    "fx_rate": 1.0,
                }
            ],
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

        symbol_sets = _watch_symbols(self.storage, "KR", trade_date)
        self.assertEqual({"AAA"}, symbol_sets["candidates"])
        self.assertEqual({"BBB"}, symbol_sets["holdings"])
        self.assertEqual({"AAA", "BBB"}, symbol_sets["watch"])

    def test_collect_breakout_intents_only_returns_valid_not_entered(self) -> None:
        trade_date = "20260303"
        self.storage.upsert_symbol_plan_rows(
            [
                {
                    "trade_date": trade_date,
                    "market": "KR",
                    "symbol": "A",
                    "name": "A",
                    "candidate_rank": 2,
                    "prev_high": 110.0,
                    "prev_low": 100.0,
                    "today_open": 100.0,
                    "breakout_price": 105.0,
                    "last_price": 110.0,
                    "is_candidate": True,
                },
                {
                    "trade_date": trade_date,
                    "market": "KR",
                    "symbol": "B",
                    "name": "B",
                    "candidate_rank": 1,
                    "prev_high": 110.0,
                    "prev_low": 100.0,
                    "today_open": 100.0,
                    "breakout_price": 104.0,
                    "last_price": 120.0,
                    "is_candidate": True,
                },
                {
                    "trade_date": trade_date,
                    "market": "KR",
                    "symbol": "C",
                    "name": "C",
                    "candidate_rank": 3,
                    "prev_high": 110.0,
                    "prev_low": 100.0,
                    "today_open": 100.0,
                    "breakout_price": 110.0,
                    "last_price": 109.0,
                    "is_candidate": True,
                },
            ]
        )
        self.storage.mark_entered_today(trade_date=trade_date, market="KR", symbol="B", entered=True)

        intents = collect_breakout_intents(cfg=object(), storage=self.storage, market="KR", trade_date=trade_date)
        self.assertEqual(["A"], [x.symbol for x in intents])
        self.assertEqual([2], [x.candidate_rank for x in intents])

    def test_execute_entry_intents_respects_fixed_budget_and_cash_limit(self) -> None:
        trade_date = "20260303"
        self.storage.upsert_day_budget(
            trade_date=trade_date,
            day_start_cash_snapshot_total=15000.0,
            per_trade_ratio=0.12,
            budget_per_trade=12000.0,
            captured_at=datetime.now().isoformat(timespec="seconds"),
            payload={"source": "unit"},
        )
        cfg = SimpleNamespace(
            strict_sync=True,
            max_positions_kr=3,
            max_positions_us=3,
            max_positions_total=6,
        )
        sync_pre = SyncResult(
            market_scope="KR",
            strict=True,
            ok=True,
            blocked=False,
            reason="",
            errors=[],
            cash_krw=15000.0,
            equity_krw=15000.0,
            positions=[],
            snapshot_id=1,
        )
        sync_post = SyncResult(
            market_scope="KR",
            strict=True,
            ok=True,
            blocked=False,
            reason="",
            errors=[],
            cash_krw=3000.0,
            equity_krw=15000.0,
            positions=[],
            snapshot_id=2,
        )
        sync_service = _FakeSyncService([sync_pre, sync_post])
        intents = [
            SignalIntent(
                market="KR",
                symbol="AAA",
                name="AAA",
                last_price=10000.0,
                breakout_price=9000.0,
                today_open=9000.0,
                candidate_rank=1,
            ),
            SignalIntent(
                market="KR",
                symbol="BBB",
                name="BBB",
                last_price=10000.0,
                breakout_price=9000.0,
                today_open=9000.0,
                candidate_rank=2,
            ),
        ]
        telegram = _DummyTelegram()
        with mock.patch(
            "systematic_alpha.trader.execution.place_market_order",
            return_value=(True, "", "ODNO1", {"rt_cd": "0"}),
        ) as mocked_place:
            outcome = execute_entry_intents(
                cfg=cfg,
                storage=self.storage,
                sync_service=sync_service,
                telegram=telegram,
                market="KR",
                trade_date=trade_date,
                intents=intents,
            )

        self.assertEqual(2, outcome.proposed)
        self.assertEqual(1, outcome.sent)
        self.assertEqual(1, outcome.rejected)
        self.assertEqual(0, outcome.skipped)
        self.assertIn("BBB:INSUFFICIENT_CASH", outcome.reject_reasons)
        self.assertEqual(1, mocked_place.call_count)

        orders = self.storage.list_orders(trade_date, "KR")
        self.assertEqual(2, len(orders))
        status_map = {row["symbol"]: row["status"] for row in orders}
        self.assertEqual("SENT", status_map["AAA"])
        self.assertEqual("REJECTED", status_map["BBB"])

    def test_us_liquidation_plan_is_relative_to_market_close(self) -> None:
        cfg = SimpleNamespace(
            us_liquidation_lead_minutes=20,
            us_liquidation_retry_minutes=5,
            us_liquidation_final_seconds=30,
        )
        ref = datetime(2026, 3, 2, 10, 0, 0, tzinfo=ET)
        plan = us_liquidation_plan(cfg, ref)
        self.assertEqual((15, 40, 0), (plan.primary_start.hour, plan.primary_start.minute, plan.primary_start.second))
        self.assertEqual((15, 55, 0), (plan.retry_time.hour, plan.retry_time.minute, plan.retry_time.second))
        self.assertEqual((15, 59, 30), (plan.final_check.hour, plan.final_check.minute, plan.final_check.second))

    def test_snapshot_budget_uses_single_cash_snapshot_ratio(self) -> None:
        trade_date = "20260303"
        cfg = SimpleNamespace(strict_sync=True, per_trade_ratio=0.12)
        sync_service = _FakeSyncService(
            [
                SyncResult(
                    market_scope="ALL",
                    strict=True,
                    ok=True,
                    blocked=False,
                    reason="",
                    errors=[],
                    cash_krw=1000000.0,
                    equity_krw=1000000.0,
                    positions=[],
                    snapshot_id=11,
                )
            ]
        )

        result = _snapshot_budget(cfg=cfg, storage=self.storage, sync=sync_service, trade_date=trade_date)
        self.assertTrue(result.get("ok"))
        self.assertEqual(120000.0, float(result["budget_per_trade"]))

        saved = self.storage.get_day_budget(trade_date)
        self.assertIsNotNone(saved)
        self.assertEqual(1000000.0, float(saved["day_start_cash_snapshot_total"]))
        self.assertEqual(0.12, float(saved["per_trade_ratio"]))
        self.assertEqual(120000.0, float(saved["budget_per_trade"]))

    def test_sync_account_blocks_us_when_any_required_exchange_fails(self) -> None:
        cfg = SimpleNamespace(
            project_root=self.root,
            broker_global_serialize=False,
            broker_global_min_interval_sec=0.0,
            us_dayornight_call_spacing_sec=0.0,
            rate_limit_retries=0,
            rate_limit_backoff_sec=0.0,
            rate_limit_backoff_max_sec=0.1,
            use_mock=True,
            us_sync_exchanges=["NYSE", "AMEX"],
            us_exchange_spacing_sec=0.0,
            us_require_all_exchanges=True,
        )
        service = AccountSyncService(cfg, self.storage)

        nyse_ok = {
            "cash_krw": 1000.0,
            "equity_krw": 1500.0,
            "positions": [
                {
                    "market": "US",
                    "symbol": "IBM",
                    "quantity": 1.0,
                    "avg_price": 100.0,
                    "market_value_krw": 130000.0,
                    "currency": "USD",
                    "fx_rate": 1300.0,
                }
            ],
        }

        with mock.patch.object(service, "_fetch_us_exchange", side_effect=[nyse_ok, RuntimeError("fail")]):
            result = service.sync_account(market_scope="US", strict=True, trade_date="20260303")

        self.assertFalse(result.ok)
        self.assertTrue(result.blocked)
        self.assertEqual("broker_fetch_failed", result.reason)
        self.assertTrue(any(err.startswith("US:") for err in result.errors))

    def test_midday_bootstrap_runs_catchup_when_precompute_and_budget_missing(self) -> None:
        trade_date = "20260303"
        cfg = SimpleNamespace(strict_sync=True, per_trade_ratio=0.12)
        telegram = _DummyTelegram()
        sync_service = _FakeSyncService(
            [
                SyncResult(
                    market_scope="ALL",
                    strict=True,
                    ok=True,
                    blocked=False,
                    reason="",
                    errors=[],
                    cash_krw=500000.0,
                    equity_krw=500000.0,
                    positions=[],
                    snapshot_id=99,
                )
            ]
        )
        now = datetime(2026, 3, 3, 13, 0, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        with mock.patch(
            "systematic_alpha.trader.cli.precompute_all_markets",
            return_value={"trade_date": trade_date, "results": []},
        ) as mocked_pre:
            with mock.patch(
                "systematic_alpha.trader.cli._compute_precompute_done_from_cache",
                side_effect=[(False, []), (True, [])],
            ):
                with mock.patch(
                    "systematic_alpha.trader.cli.precompute_window",
                    return_value=(now.replace(hour=8, minute=5), now.replace(hour=8, minute=33)),
                ):
                    with mock.patch(
                        "systematic_alpha.trader.cli.budget_snapshot_time",
                        return_value=now.replace(hour=8, minute=40),
                    ):
                        with mock.patch(
                            "systematic_alpha.trader.cli.is_market_open",
                            side_effect=lambda market: str(market).upper() == "KR",
                        ):
                            result = _ensure_daily_bootstrap(
                                cfg=cfg,
                                storage=self.storage,
                                sync=sync_service,
                                telegram=telegram,
                                now_kst_dt=now,
                                trade_date=trade_date,
                            )

        self.assertTrue(result["ran_precompute"])
        self.assertTrue(result["ran_budget_snapshot"])
        mocked_pre.assert_called_once()
        self.assertEqual("1", str(self.storage.get_meta(f"precompute_done:{trade_date}", "0")))
        self.assertIsNotNone(self.storage.get_day_budget(trade_date))

    def test_profile_paths_are_separated_for_test_profile(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "TRADER_PROFILE": "test",
                "TELEGRAM_ENABLED": "0",
                "TELEGRAM_BOT_TOKEN": "",
                "TELEGRAM_CHAT_ID": "",
            },
            clear=False,
        ):
            cfg = load_trader_config(self.root)
        self.assertEqual("test", cfg.profile)
        self.assertEqual(self.root / "state" / "trader_test", cfg.state_dir)
        self.assertEqual(self.root / "out" / "trader_test", cfg.out_dir)
        self.assertEqual(self.root / "logs" / "trader_test", cfg.logs_dir)

    def test_sync_api_error_text_treats_benign_kis_notices_as_success(self) -> None:
        payload_with_notice_and_output = {
            "msg_cd": "20310000",
            "msg1": "모의투자 조회가 완료되었습니다.",
            "output2": {"dnca_tot_amt": "1000000"},
        }
        payload_no_data_notice = {
            "msg_cd": "70070000",
            "msg1": "모의투자 조회할 내역(자료)이 없습니다.",
        }
        payload_real_error = {
            "rt_cd": "1",
            "msg_cd": "EGW00201",
            "msg1": "초당 거래건수를 초과하였습니다.",
        }

        self.assertEqual("", _api_error_text(payload_with_notice_and_output))
        self.assertEqual("", _api_error_text(payload_no_data_notice))
        self.assertIn("rt_cd=1", _api_error_text(payload_real_error))


if __name__ == "__main__":
    unittest.main()
