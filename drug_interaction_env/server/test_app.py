import unittest
from fastapi.testclient import TestClient
from drug_interaction_env.server.app import app

class TestAppEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_reset_endpoint(self):
        response = self.client.post("/reset", json={"task_level": "easy"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("patient_id", data)
        self.assertIn("medications", data)
        self.assertIn("steps_remaining", data)
        self.assertEqual(data["flags_raised_so_far"], [])

    def test_step_endpoint(self):
        self.client.post("/reset", json={"task_level": "easy"})
        action = {
            "action_type": "flag_interaction",
            "drug_a": "warfarin",
            "drug_b": "aspirin",
            "severity": "severe",
            "suggested_action": "replace_drug"
        }
        response = self.client.post("/step", json=action)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("observation", data)
        self.assertIn("reward", data)
        self.assertIn("done", data)
        self.assertIn("state", data)
        self.assertTrue(data["done"])
        self.assertAlmostEqual(data["reward"], 0.8)

    def test_state_endpoint(self):
        self.client.post("/reset", json={"task_level": "easy"})
        response = self.client.get("/state")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["task_level"], "easy")
        self.assertIn("episode_score", data)

if __name__ == "__main__":
    unittest.main()
