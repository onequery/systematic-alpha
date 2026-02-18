import json
import tempfile
import unittest
from pathlib import Path

from systematic_alpha.agent_lab.orchestrator import AgentLabOrchestrator


class AgentLabOrchestratorTests(unittest.TestCase):
    def test_init_ingest_propose_approve_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            result_dir = root / "out" / "kr" / run_date / "results"
            result_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "market": "KR",
                "generated_at": "2026-02-18T09:10:00+09:00",
                "signal_valid": True,
                "invalid_reason": None,
                "final": [],
                "all_ranked": [
                    {
                        "rank": 1,
                        "code": "005930",
                        "name": "SAMSUNG",
                        "score": 7,
                        "max_score": 8,
                        "recommendation_score": 87.5,
                        "metrics": {"latest_price": 70000},
                    },
                    {
                        "rank": 2,
                        "code": "000660",
                        "name": "SKHYNIX",
                        "score": 6,
                        "max_score": 8,
                        "recommendation_score": 75.0,
                        "metrics": {"latest_price": 120000},
                    },
                    {
                        "rank": 3,
                        "code": "035420",
                        "name": "NAVER",
                        "score": 6,
                        "max_score": 8,
                        "recommendation_score": 75.0,
                        "metrics": {"latest_price": 200000},
                    },
                    {
                        "rank": 4,
                        "code": "051910",
                        "name": "LGCHEM",
                        "score": 5,
                        "max_score": 8,
                        "recommendation_score": 62.5,
                        "metrics": {"latest_price": 350000},
                    },
                    {
                        "rank": 5,
                        "code": "005380",
                        "name": "HYUNDAI",
                        "score": 5,
                        "max_score": 8,
                        "recommendation_score": 62.5,
                        "metrics": {"latest_price": 180000},
                    },
                ],
            }
            out_json = result_dir / "kr_daily_20260218_091001.json"
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            orch = AgentLabOrchestrator(project_root=root)
            try:
                init_payload = orch.init_lab(capital_krw=10_000_000, agents=3)
                self.assertEqual(3, len(init_payload["agents"]))

                ingest_payload = orch.ingest_session(market="KR", yyyymmdd=run_date)
                self.assertEqual("SIGNAL_OK", ingest_payload["status_code"])

                proposal_payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                self.assertEqual(3, len(proposal_payload["proposals"]))

                pending = [x for x in proposal_payload["proposals"] if x["status"] == "PENDING_APPROVAL"]
                self.assertGreaterEqual(len(pending), 1)

                approval_payload = orch.approve_orders(str(pending[0]["proposal_id"]), approved_by="unittest")
                self.assertEqual("EXECUTED", approval_payload["status"])

                review_payload = orch.daily_review(run_date)
                self.assertEqual(run_date, review_payload["date"])
                self.assertEqual(3, len(review_payload["rows"]))
            finally:
                orch.close()


if __name__ == "__main__":
    unittest.main()

