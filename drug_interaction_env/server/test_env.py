import unittest
from drug_interaction_env.server.drug_interaction_environment import DrugInteractionEnvironment
from drug_interaction_env.server.drug_database import DRUG_INTERACTIONS

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = DrugInteractionEnvironment()
        self.env.reset("easy")  # Medications: warfarin, aspirin, losartan, amlodipine, gabapentin, pantoprazole

    def test_validate_valid_pair(self):
        action = {"drug_a": "warfarin", "drug_b": "aspirin"}
        self.assertTrue(self.env.validate(action))
        
    def test_validate_invalid_drug(self):
        # Fake drug
        action = {"drug_a": "warfarin", "drug_b": "fake_drug"}
        self.assertFalse(self.env.validate(action))

    def test_validate_phantom_pair(self):
        # Both exist but don't interact in DB
        action = {"drug_a": "losartan", "drug_b": "amlodipine"}
        self.assertFalse(self.env.validate(action))

    def test_validate_duplicate(self):
        action = {"drug_a": "warfarin", "drug_b": "aspirin"}
        self.assertTrue(self.env.validate(action))
        self.assertIsNone(self.env.validate(action))  

    def test_calculate_reward_perfect(self):
        key = ("aspirin", "warfarin") 
        reward = self.env.calculate_reward(key, "severe", "replace_drug")
     
        self.assertAlmostEqual(reward, 0.8)
        self.assertIn(key, self.env.perfectly_completed_pairs)
        self.assertIn(key, self.env.identified_pairs)

    def test_calculate_reward_wrong_severity_severe(self):
        key = ("aspirin", "warfarin")
        reward = self.env.calculate_reward(key, "moderate", "replace_drug")
     
        self.assertAlmostEqual(reward, 0.4)
        self.assertNotIn(key, self.env.perfectly_completed_pairs)

    def test_calculate_reward_wrong_action(self):
        key = ("aspirin", "warfarin")
        reward = self.env.calculate_reward(key, "severe", "monitor")
      
        self.assertAlmostEqual(reward, 0.5)

    def test_complete_episode(self):
        self.env.reset("easy")
        obs, reward, done, state = self.env.step({
            "action_type": "flag_interaction",
            "drug_a": "warfarin",
            "drug_b": "aspirin",
            "severity": "severe",
            "suggested_action": "replace_drug"
        })
        self.assertAlmostEqual(reward, 0.8)
        self.assertTrue(done)
        self.assertAlmostEqual(self.env.get_episode_score(), 1.0)

    def test_done_action_penalties(self):
        self.env.reset("easy")
        obs, reward, done, state = self.env.step({"action_type": "DONE"})
        # 1 severe missed -> penalty = -0.4
        self.assertAlmostEqual(reward, -0.4)
        self.assertTrue(done)
        self.assertAlmostEqual(self.env.get_episode_score(), 0.0) 

    def test_medium_episode(self):
        self.env.reset("medium")
        

        obs, reward, done, state = self.env.step({
            "action_type": "flag_interaction",
            "drug_a": "clopidogrel",
            "drug_b": "omeprazole",
            "severity": "severe",
            "suggested_action": "replace_drug"
        })
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.8)
        
        obs, reward, done, state = self.env.step({
            "action_type": "flag_interaction",
            "drug_a": "ciprofloxacin",
            "drug_b": "theophylline",
            "severity": "moderate",
            "suggested_action": "reduce_dose"
        })
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.8)


        obs, reward, done, state = self.env.step({
            "action_type": "flag_interaction",
            "drug_a": "levothyroxine",
            "drug_b": "calcium",
            "severity": "mild",
            "suggested_action": "monitor"
        })
        
        self.assertTrue(done)
        self.assertAlmostEqual(self.env.get_episode_score(), 1.0)

    def test_hard_episode(self):
        self.env.reset("hard")
        
        actions = [
            ("amiodarone", "simvastatin", "severe", "reduce_dose"),
            ("amiodarone", "digoxin", "severe", "reduce_dose"),
            ("ssri", "tramadol", "severe", "replace_drug"),
            ("digoxin", "furosemide", "moderate", "monitor"),
            ("levothyroxine", "calcium", "mild", "monitor"),
        ]
        
        for i, (da, db, sev, act) in enumerate(actions):
            obs, reward, done, state = self.env.step({
                "action_type": "flag_interaction",
                "drug_a": da,
                "drug_b": db,
                "severity": sev,
                "suggested_action": act
            })
            self.assertAlmostEqual(reward, 0.8)
            if i < len(actions) - 1:
                self.assertFalse(done)
            else:
                self.assertTrue(done)
        
        self.assertAlmostEqual(self.env.get_episode_score(), 1.0)

if __name__ == "__main__":
    unittest.main()
