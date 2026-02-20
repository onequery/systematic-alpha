import json
import tempfile
import unittest
from pathlib import Path

from systematic_alpha.agent_lab.orchestrator import AgentLabOrchestrator


def _sample_signal_payload(market: str) -> dict:
    return {
        "market": market,
        "generated_at": "2026-02-18T09:10:00+09:00",
        "signal_valid": True,
        "invalid_reason": None,
        "final": [],
        "all_ranked": [
            {
                "rank": 1,
                "code": "005930" if market == "KR" else "AAPL",
                "name": "SAMSUNG" if market == "KR" else "APPLE",
                "score": 7,
                "max_score": 8,
                "recommendation_score": 87.5,
                "metrics": {"latest_price": 70000 if market == "KR" else 190.0},
            },
            {
                "rank": 2,
                "code": "000660" if market == "KR" else "MSFT",
                "name": "SKHYNIX" if market == "KR" else "MICROSOFT",
                "score": 6,
                "max_score": 8,
                "recommendation_score": 75.0,
                "metrics": {"latest_price": 120000 if market == "KR" else 410.0},
            },
            {
                "rank": 3,
                "code": "035420" if market == "KR" else "NVDA",
                "name": "NAVER" if market == "KR" else "NVIDIA",
                "score": 6,
                "max_score": 8,
                "recommendation_score": 73.0,
                "metrics": {"latest_price": 200000 if market == "KR" else 720.0},
            },
            {
                "rank": 4,
                "code": "051910" if market == "KR" else "AMZN",
                "name": "LGCHEM" if market == "KR" else "AMAZON",
                "score": 5,
                "max_score": 8,
                "recommendation_score": 62.5,
                "metrics": {"latest_price": 350000 if market == "KR" else 170.0},
            },
            {
                "rank": 5,
                "code": "005380" if market == "KR" else "META",
                "name": "HYUNDAI" if market == "KR" else "META",
                "score": 5,
                "max_score": 8,
                "recommendation_score": 61.5,
                "metrics": {"latest_price": 180000 if market == "KR" else 490.0},
            },
        ],
    }


def _write_signal_file(root: Path, market: str, run_date: str) -> None:
    market_lc = market.lower()
    result_dir = root / "out" / market_lc / run_date / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = _sample_signal_payload(market)
    out_json = result_dir / f"{market_lc}_daily_{run_date}_091001.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AgentLabOrchestratorTests(unittest.TestCase):
    def test_init_ingest_propose_auto_execute_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)

            orch = AgentLabOrchestrator(project_root=root)
            try:
                init_payload = orch.init_lab(capital_krw=10_000_000, agents=3)
                self.assertEqual(3, len(init_payload["agents"]))

                ingest_payload = orch.ingest_session(market="KR", yyyymmdd=run_date)
                self.assertEqual("SIGNAL_OK", ingest_payload["status_code"])

                proposal_payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                self.assertEqual(3, len(proposal_payload["proposals"]))

                statuses = {str(x.get("status", "")) for x in proposal_payload["proposals"]}
                self.assertIn("EXECUTED", statuses)

                review_payload = orch.daily_review(run_date)
                self.assertEqual(run_date, review_payload["date"])
                self.assertEqual(3, len(review_payload["rows"]))
            finally:
                orch.close()

    def test_cross_market_plan_caps_kr_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)

            orch = AgentLabOrchestrator(project_root=root)
            try:
                orch.init_lab(capital_krw=10_000_000, agents=1)
                orch.ingest_session(market="KR", yyyymmdd=run_date)
                proposal_payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                self.assertEqual(1, len(proposal_payload["proposals"]))

                proposal = proposal_payload["proposals"][0]
                plan = proposal.get("cross_market_plan", {})
                self.assertIn("target_weights", plan)
                self.assertIn("KR", plan["target_weights"])
                self.assertIn("US", plan["target_weights"])

                budget_cap = float(plan.get("buy_budget_cap_krw", 0.0) or 0.0)
                self.assertGreater(budget_cap, 0.0)
                self.assertLessEqual(budget_cap, 6_700_000.0)

                buy_notional = 0.0
                for order in list(proposal.get("orders") or []):
                    if str(order.get("side", "")).upper() != "BUY":
                        continue
                    buy_notional += float(order.get("quantity", 0.0) or 0.0) * float(order.get("reference_price", 0.0) or 0.0)
                self.assertLessEqual(buy_notional, budget_cap + 1.0)
            finally:
                orch.close()


if __name__ == "__main__":
    unittest.main()
