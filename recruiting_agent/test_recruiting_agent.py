import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch


recruiting_agent_module = importlib.import_module("recruiting_agent.recruiting_agent")


class RecruiterInitializationTests(unittest.TestCase):
    def invoke_with_result(self, user_id):
        result = {"messages": [SimpleNamespace(content="done")]}
        with patch.object(recruiting_agent_module.recruiting_agent, "invoke", return_value=result) as invoke:
            recruiting_agent_module.run_agent("Build a profile", user_id=user_id)
        return invoke.call_args

    def test_resolved_identity_is_available_before_agent_invocation(self):
        call = self.invoke_with_result("recruiter_amills")
        agent_input = call.args[0]
        metadata = call.kwargs["config"]["metadata"]
        identity_message = agent_input["messages"][0]

        self.assertEqual(identity_message["role"], "system")
        self.assertEqual(
            json.loads(identity_message["content"].removeprefix("Authenticated recruiter identity: ")),
            {
                "recruiter_id": "recruiter_amills",
                "name": "Ava Mills",
                "email": "ava.mills@northpoint.com",
            },
        )
        self.assertEqual(agent_input["messages"][1]["role"], "user")
        self.assertEqual(metadata["recruiter"]["recruiter_id"], "recruiter_amills")

    def test_missing_and_unrecognized_ids_remain_unavailable(self):
        for user_id in (None, "unknown_recruiter"):
            with self.subTest(user_id=user_id):
                call = self.invoke_with_result(user_id)
                metadata = call.kwargs["config"]["metadata"]
                self.assertEqual(
                    call.args[0]["messages"][0]["content"],
                    "Authenticated recruiter identity: unavailable.",
                )
                self.assertIsNone(metadata["recruiter"])

    def test_lookup_tool_resolves_the_authenticated_recruiter(self):
        result = recruiting_agent_module.get_current_recruiter.invoke(
            {}, config={"metadata": {"user_id": "recruiter_amills"}}
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["recruiter"]["recruiter_id"], "recruiter_amills")

    def test_email_tool_uses_the_resolved_recruiter(self):
        result = recruiting_agent_module.send_candidate_email.invoke(
            {
                "candidate": {"name": "Candidate", "email": "candidate@example.com"},
                "subject": "Interview",
                "body": "Let's talk.",
            },
            config={"metadata": {"user_id": "recruiter_amills"}},
        )

        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["from_name"], "Ava Mills")

    def test_invalid_identity_is_not_used_by_recruiter_tools(self):
        config = {"metadata": {"user_id": "unknown_recruiter"}}
        recruiter = recruiting_agent_module.get_current_recruiter.invoke({}, config=config)
        email = recruiting_agent_module.send_candidate_email.invoke(
            {
                "candidate": {"name": "Candidate", "email": "candidate@example.com"},
                "subject": "Interview",
                "body": "Let's talk.",
            },
            config=config,
        )

        self.assertFalse(recruiter["found"])
        self.assertIsNone(email["from"])
        self.assertIsNone(email["from_name"])


if __name__ == "__main__":
    unittest.main()
