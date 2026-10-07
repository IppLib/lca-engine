import copy
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
package = types.ModuleType("recruiting_agent")
package.__path__ = [os.path.dirname(__file__)]
sys.modules["recruiting_agent"] = package

from recruiting_agent import data_service
from recruiting_agent import recruiting_agent
from recruiting_agent.recruiting_records import CANDIDATES


class SkillConsistencyTests(unittest.TestCase):
    candidate_id = "CAND-39002"

    def setUp(self):
        self.original_candidate = copy.deepcopy(CANDIDATES[self.candidate_id])
        data_service._PROFILES.clear()

    def tearDown(self):
        CANDIDATES[self.candidate_id] = self.original_candidate
        data_service._PROFILES.clear()

    def test_skill_write_updates_source_and_cached_profile(self):
        profile = recruiting_agent.build_candidate_profile.func(self.candidate_id)["candidate_profile"]

        result = data_service.add_candidate_skill(self.candidate_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_skills(self.candidate_id))
        self.assertIn("Terraform", data_service.get_profile_from_db(self.candidate_id)["candidate_profile"]["skills"])
        self.assertIs(profile, data_service.get_profile_from_db(self.candidate_id)["candidate_profile"])

    def test_build_profile_after_skill_write_reads_new_skill(self):
        data_service.add_candidate_skill(self.candidate_id, "Terraform")

        profile = recruiting_agent.build_candidate_profile.func(self.candidate_id)["candidate_profile"]

        self.assertIn("Terraform", profile["skills"])

    def test_score_preserves_supplied_skills_and_merges_fetched_skills(self):
        candidate_profile = {
            "candidate_id": self.candidate_id,
            "skills": ["CallerOnly"],
        }
        job_description = {
            "required_skills": ["CallerOnly"],
            "min_years_experience": 1,
            "description": "A role requiring the supplied skill.",
        }
        score = SimpleNamespace(model_dump=lambda: {"score": 90})

        with patch.object(recruiting_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = score
            recruiting_agent.score_candidate.func(candidate_profile, job_description)

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(user_message.split("\n\nCandidate profile:\n", 1)[1])
        self.assertIn("CallerOnly", scored_profile["skills"])
        self.assertIn("Go", scored_profile["skills"])

    def test_agent_prompt_flags_contradictory_score_justification(self):
        self.assertIn("explicitly flag the inconsistency", recruiting_agent.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
