import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from systematic_alpha.agent_lab.paper_broker import PaperBroker
from systematic_alpha.agent_lab.storage import AgentLabStorage


class PaperBrokerUsSnapshotTests(unittest.TestCase):
    def test_us_exchange_candidates_default_to_nyse_amex(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "agent_lab.sqlite"
            storage = AgentLabStorage(db_path)
            broker = PaperBroker(storage)
            try:
                with patch.dict("os.environ", {}, clear=True):
                    out = broker._exchange_candidates("US")
                self.assertEqual(["NYSE", "AMEX"], out)
            finally:
                storage.close()

    def test_us_snapshot_requires_all_exchanges_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "agent_lab.sqlite"
            storage = AgentLabStorage(db_path)
            broker = PaperBroker(storage)
            try:
                with patch.dict(
                    "os.environ",
                    {
                        "AGENT_LAB_US_BALANCE_INCLUDE_ALL_MARKET": "0",
                        "AGENT_LAB_US_BALANCE_REQUIRE_ALL_EXCHANGES": "1",
                        "AGENT_LAB_US_BALANCE_EXCHANGE_SPACING_SEC": "0",
                        "AGENT_LAB_US_BALANCE_RATE_LIMIT_COOLDOWN_SEC": "0",
                    },
                    clear=False,
                ):
                    with patch.object(broker, "_exchange_candidates", return_value=["NASD", "NYSE", "AMEX"]):
                        with patch.object(broker, "_get_broker", side_effect=lambda _m, ex="": ex):
                            with patch.object(
                                broker,
                                "_fetch_us_balance_with_fallback",
                                side_effect=lambda ex: {"exchange": ex},
                            ):
                                with patch.object(
                                    broker,
                                    "_parse_balance_oversea",
                                    side_effect=lambda payload: {
                                        "cash_krw": 1000.0,
                                        "equity_krw": 1200.0,
                                        "positions": [
                                            {
                                                "market": "US",
                                                "symbol": f"{str(payload.get('exchange', '')).upper()}_SYMBOL",
                                                "quantity": 1.0,
                                                "avg_price": 1.0,
                                                "market_value_krw": 1.0,
                                                "currency": "USD",
                                                "fx_rate": 1.0,
                                            }
                                        ],
                                    },
                                ):
                                    out = broker._fetch_us_snapshot_all_exchanges()
                self.assertTrue(bool(out.get("all_required_exchanges_ok")))
                self.assertEqual(["AMEX", "NASD", "NYSE"], sorted(list(out.get("successful_required", []))))
            finally:
                storage.close()

    def test_us_snapshot_fails_when_one_exchange_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "agent_lab.sqlite"
            storage = AgentLabStorage(db_path)
            broker = PaperBroker(storage)
            try:
                with patch.dict(
                    "os.environ",
                    {
                        "AGENT_LAB_US_BALANCE_INCLUDE_ALL_MARKET": "0",
                        "AGENT_LAB_US_BALANCE_REQUIRE_ALL_EXCHANGES": "1",
                        "AGENT_LAB_US_BALANCE_EXCHANGE_SPACING_SEC": "0",
                        "AGENT_LAB_US_BALANCE_RATE_LIMIT_COOLDOWN_SEC": "1.23",
                    },
                    clear=False,
                ):
                    with patch.object(broker, "_exchange_candidates", return_value=["NASD", "NYSE", "AMEX"]):
                        with patch.object(broker, "_get_broker", side_effect=lambda _m, ex="": ex):
                            def _fetch_side_effect(ex: str):
                                if str(ex).upper() == "AMEX":
                                    raise RuntimeError(
                                        "rt_cd=1, msg_cd=EGW00201, msg1=초당 거래건수를 초과하였습니다."
                                    )
                                return {"exchange": ex}

                            with patch.object(
                                broker,
                                "_fetch_us_balance_with_fallback",
                                side_effect=_fetch_side_effect,
                            ):
                                with patch.object(
                                    broker,
                                    "_parse_balance_oversea",
                                    side_effect=lambda payload: {
                                        "cash_krw": 1000.0,
                                        "equity_krw": 1200.0,
                                        "positions": [
                                            {
                                                "market": "US",
                                                "symbol": f"{str(payload.get('exchange', '')).upper()}_SYMBOL",
                                                "quantity": 1.0,
                                                "avg_price": 1.0,
                                                "market_value_krw": 1.0,
                                                "currency": "USD",
                                                "fx_rate": 1.0,
                                            }
                                        ],
                                    },
                                ):
                                    with patch("systematic_alpha.agent_lab.paper_broker.time.sleep") as mocked_sleep:
                                        with self.assertRaises(RuntimeError) as exc:
                                            broker._fetch_us_snapshot_all_exchanges()
                self.assertIn("us_incomplete_snapshot", str(exc.exception))
                self.assertIn("AMEX", str(exc.exception))
                self.assertTrue(any(abs(float(call.args[0]) - 1.23) < 1e-6 for call in mocked_sleep.call_args_list))
            finally:
                storage.close()

    def test_fetch_account_snapshot_us_fail_close_on_incomplete_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "agent_lab.sqlite"
            storage = AgentLabStorage(db_path)
            broker = PaperBroker(storage)
            try:
                with patch.object(
                    broker,
                    "_fetch_us_snapshot_all_exchanges",
                    side_effect=RuntimeError("us_incomplete_snapshot:{\"missing_required\": [\"AMEX\"]}"),
                ):
                    out = broker.fetch_account_snapshot("US")
                self.assertFalse(bool(out.get("ok")))
                self.assertEqual("US", str(out.get("market_scope")))
                self.assertEqual({}, out.get("markets"))
                errs = list(out.get("errors", []) or [])
                self.assertTrue(any(str(e).startswith("US:") for e in errs))
            finally:
                storage.close()


if __name__ == "__main__":
    unittest.main()
