import unittest

from systematic_alpha.agent_lab.risk_engine import RiskEngine
from systematic_alpha.agent_lab.schemas import STATUS_MARKET_CLOSED, STATUS_SIGNAL_OK


class RiskEngineTests(unittest.TestCase):
    def test_blocked_when_market_closed(self) -> None:
        engine = RiskEngine()
        decision = engine.evaluate(
            status_code=STATUS_MARKET_CLOSED,
            allocated_capital_krw=3_333_333,
            available_cash_krw=3_333_333,
            day_return_pct=0.0,
            week_return_pct=0.0,
            current_exposure_krw=0.0,
            orders=[],
        )
        self.assertFalse(decision.allowed)
        self.assertIn("blocked_by_status", decision.blocked_reason)

    def test_blocks_when_day_loss_limit_hit(self) -> None:
        engine = RiskEngine()
        decision = engine.evaluate(
            status_code=STATUS_SIGNAL_OK,
            allocated_capital_krw=3_333_333,
            available_cash_krw=3_333_333,
            day_return_pct=-0.03,
            week_return_pct=0.0,
            current_exposure_krw=0.0,
            orders=[],
        )
        self.assertFalse(decision.allowed)
        self.assertIn("day_loss_limit", decision.blocked_reason)

    def test_accepts_single_valid_order(self) -> None:
        engine = RiskEngine()
        order = {
            "symbol": "005930",
            "side": "BUY",
            "quantity": 10,
            "reference_price": 70000,
        }
        decision = engine.evaluate(
            status_code=STATUS_SIGNAL_OK,
            allocated_capital_krw=3_333_333,
            available_cash_krw=3_333_333,
            day_return_pct=0.01,
            week_return_pct=0.02,
            current_exposure_krw=0.0,
            orders=[order],
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(1, len(decision.accepted_orders))
        self.assertEqual("005930", decision.accepted_orders[0]["symbol"])


if __name__ == "__main__":
    unittest.main()
