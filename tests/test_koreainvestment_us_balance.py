import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "mojito" / "mojito" / "koreainvestment.py"
_SPEC = importlib.util.spec_from_file_location("local_koreainvestment", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load module spec: {_MODULE_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
KoreaInvestment = _MOD.KoreaInvestment


class _DummyResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class KoreaInvestmentUsBalanceTests(unittest.TestCase):
    @staticmethod
    def _make_client() -> KoreaInvestment:
        with patch.object(KoreaInvestment, "check_access_token", return_value=True):
            with patch.object(
                KoreaInvestment,
                "load_access_token",
                side_effect=lambda self: setattr(self, "access_token", "Bearer unit-test"),
                autospec=True,
            ):
                return KoreaInvestment(
                    api_key="k",
                    api_secret="s",
                    acc_no="12345678-01",
                    exchange="나스닥",
                    mock=True,
                )

    def test_fetch_balance_oversea_calls_dayornight_every_request(self) -> None:
        client = self._make_client()
        with patch.dict("os.environ", {"TRADER_US_DAYORNIGHT_CALL_SPACING_SEC": "0"}, clear=False):
            with patch.object(client, "fetch_oversea_day_night", return_value={"output": {"PSBL_YN": "N"}}) as m_day:
                with patch(f"{_MOD.__name__}.requests.get") as m_get:
                    m_get.return_value = _DummyResponse(
                        {"rt_cd": "0", "output1": [], "output2": []},
                        {"tr_cont": ""},
                    )
                    client.fetch_balance_oversea()
                    client.fetch_balance_oversea()
        self.assertEqual(2, m_day.call_count)
        self.assertEqual(2, m_get.call_count)

    def test_fetch_balance_oversea_applies_dayornight_to_balance_spacing(self) -> None:
        client = self._make_client()
        events = []

        def _daynight():
            events.append("daynight")
            return {"output": {"PSBL_YN": "Y"}}

        def _sleep(sec: float):
            events.append(f"sleep:{sec}")

        def _get(*_args, **_kwargs):
            events.append("balance_get")
            return _DummyResponse({"rt_cd": "0", "output1": [], "output2": []}, {"tr_cont": ""})

        with patch.dict("os.environ", {"TRADER_US_DAYORNIGHT_CALL_SPACING_SEC": "0.6"}, clear=False):
            with patch.object(client, "fetch_oversea_day_night", side_effect=_daynight):
                with patch(f"{_MOD.__name__}.time.sleep", side_effect=_sleep) as m_sleep:
                    with patch(f"{_MOD.__name__}.requests.get", side_effect=_get):
                        client.fetch_balance_oversea()

        self.assertGreaterEqual(m_sleep.call_count, 1)
        self.assertTrue(any(abs(float(c.args[0]) - 0.6) < 1e-6 for c in m_sleep.call_args_list))
        self.assertLess(events.index("daynight"), events.index("balance_get"))
        sleep_events = [e for e in events if e.startswith("sleep:")]
        self.assertTrue(sleep_events)
        self.assertLess(events.index(sleep_events[0]), events.index("balance_get"))


if __name__ == "__main__":
    unittest.main()
