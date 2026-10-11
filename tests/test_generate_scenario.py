"""
tests/test_generate_scenario.py
Unit tests for POST /ai/assistant/generate-scenario.

LLM calls and Supabase calls are mocked so the suite runs offline in CI
without any real API keys or database connections.
"""

import json
import types
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Minimal env so ml.supabase_client doesn't crash at import ─────────────
import os
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from main import app  # noqa: E402  (env must be set before this import)

client = TestClient(app)

# ── Helpers ──────────────────────────────────────────────────────────────────

PATIENT_ID = "22222222-2222-2222-2222-222222222201"

VALID_LLM_JSON = {
    "title": "Sundowning Agitation at Dusk",
    "scenario_text": (
        "Lakshmi becomes increasingly agitated around 6 PM, asking to go home "
        "repeatedly even though she is already home. She begins to pace and looks "
        "distressed. What is the most supportive response?"
    ),
    "difficulty": "medium",
    "category": "dementia_care",
    "options": [
        {
            "option_text": "Calmly acknowledge her feelings and gently redirect with a familiar activity",
            "is_correct": True,
            "explanation": (
                "Validation and redirection reduce distress without increasing "
                "confusion or agitation in dementia care."
            ),
        },
        {
            "option_text": "Firmly remind her that she is at home and there is nothing to worry about",
            "is_correct": False,
            "explanation": (
                "Contradicting or correcting can increase agitation because the "
                "patient cannot process the correction."
            ),
        },
        {
            "option_text": "Leave her alone until she calms down",
            "is_correct": False,
            "explanation": (
                "Leaving a distressed person alone is unsafe and increases anxiety."
            ),
        },
    ],
}


def _make_supabase_mock(patient_found: bool = True, persist_ok: bool = True):
    """Build a mock Supabase client with chained query methods."""
    sb = MagicMock()

    # patient row
    patient_row = (
        {
            "full_name": "Lakshmi Krishnan",
            "condition_type": "Dementia",
            "medical_notes": "Mild cognitive decline.",
        }
        if patient_found
        else None
    )

    # Generate stable IDs for scenario card / options
    card_id = str(uuid.uuid4())
    option_id = str(uuid.uuid4())

    def _from(table):
        q = MagicMock()
        q.select.return_value = q
        q.eq.return_value = q
        q.order.return_value = q
        q.limit.return_value = q
        q.single.return_value = q
        q.insert.return_value = q

        if table == "patient":
            q.execute.return_value = MagicMock(data=patient_row)
        elif table == "time_event":
            q.execute.return_value = MagicMock(data=[])
        elif table == "daily_care_log":
            q.execute.return_value = MagicMock(data=[])
        elif table == "scenario_card":
            card_data = (
                [
                    {
                        "scenario_id": card_id,
                        "patient_id": PATIENT_ID,
                        "title": VALID_LLM_JSON["title"],
                        "scenario_text": VALID_LLM_JSON["scenario_text"],
                        "difficulty": "medium",
                        "category": "dementia_care",
                        "is_ai_generated": True,
                    }
                ]
                if persist_ok
                else None
            )
            q.execute.return_value = MagicMock(data=card_data)
        elif table == "scenario_option":
            opt_data = (
                [
                    {
                        "option_id": option_id,
                        "scenario_id": card_id,
                        "option_text": VALID_LLM_JSON["options"][0]["option_text"],
                        "is_correct": True,
                        "explanation": VALID_LLM_JSON["options"][0]["explanation"],
                    }
                ]
                if persist_ok
                else None
            )
            q.execute.return_value = MagicMock(data=opt_data)
        else:
            q.execute.return_value = MagicMock(data=[])
        return q

    sb.from_ = _from
    return sb


def _make_llm_mock(raw_text: str):
    """Return a mock Gemini client whose generate_content returns raw_text."""
    llm = MagicMock()
    response = MagicMock()
    response.text = raw_text
    llm.models.generate_content.return_value = response
    return llm


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGenerateScenario:
    """Happy-path and error-path tests for POST /ai/assistant/generate-scenario."""

    def test_happy_path_returns_card_and_options(self):
        """Full green-path: patient found, LLM returns valid JSON, DB persists rows."""
        sb_mock = _make_supabase_mock()
        llm_mock = _make_llm_mock(json.dumps(VALID_LLM_JSON))

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "card" in body
        assert "options" in body
        card = body["card"]
        assert card["patient_id"] == PATIENT_ID
        assert card["is_ai_generated"] is True
        assert card["title"] == VALID_LLM_JSON["title"]
        assert isinstance(body["options"], list)
        assert len(body["options"]) >= 1
        # Exactly one correct option
        correct = [o for o in body["options"] if o["is_correct"]]
        assert len(correct) == 1

    def test_llm_response_with_markdown_fences_is_handled(self):
        """Model wraps JSON in ```json fences — should still parse cleanly."""
        fenced = "```json\n" + json.dumps(VALID_LLM_JSON) + "\n```"
        sb_mock = _make_supabase_mock()
        llm_mock = _make_llm_mock(fenced)

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )

        assert resp.status_code == 200, resp.text

    def test_patient_not_found_returns_404(self):
        sb_mock = _make_supabase_mock(patient_found=False)
        llm_mock = _make_llm_mock(json.dumps(VALID_LLM_JSON))

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": "00000000-0000-0000-0000-000000000000"},
            )

        assert resp.status_code == 404

    def test_invalid_llm_json_returns_502(self):
        """LLM returns garbage — should get 502, not 500."""
        sb_mock = _make_supabase_mock()
        llm_mock = _make_llm_mock("This is not JSON at all.")

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )

        assert resp.status_code == 502

    def test_missing_llm_fields_returns_502(self):
        """LLM omits required field 'options' — should get 502."""
        bad_json = {k: v for k, v in VALID_LLM_JSON.items() if k != "options"}
        sb_mock = _make_supabase_mock()
        llm_mock = _make_llm_mock(json.dumps(bad_json))

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )

        assert resp.status_code == 502

    def test_multiple_correct_options_returns_502(self):
        """LLM marks 2 options correct — validation must reject it."""
        bad = json.loads(json.dumps(VALID_LLM_JSON))  # deep copy
        bad["options"][1]["is_correct"] = True  # second option also correct
        sb_mock = _make_supabase_mock()
        llm_mock = _make_llm_mock(json.dumps(bad))

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )

        assert resp.status_code == 502

    def test_no_supabase_config_returns_503(self):
        """Server has no Supabase credentials → 503."""
        with patch("ai.ai_router._get_supabase", return_value=None):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )
        assert resp.status_code == 503

    def test_no_gemini_key_returns_503(self):
        """Server has Supabase but no Gemini key → 503."""
        sb_mock = _make_supabase_mock()
        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=None),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )
        assert resp.status_code == 503

    def test_llm_exception_returns_500(self):
        """LLM call raises an unexpected exception → 500."""
        sb_mock = _make_supabase_mock()
        llm_mock = MagicMock()
        llm_mock.models.generate_content.side_effect = RuntimeError("Network timeout")

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )
        assert resp.status_code == 500

    def test_system_prompt_contains_safety_rules(self):
        """Ensure the SCENARIO_SYSTEM_INSTRUCTION has required SRS safety text."""
        from ai.ai_router import SCENARIO_SYSTEM_INSTRUCTION

        assert "NEVER provide medical diagnoses" in SCENARIO_SYSTEM_INSTRUCTION
        assert "NEVER give specific medication dosing instructions" in SCENARIO_SYSTEM_INSTRUCTION
        assert "SUPPORTIVE and EDUCATIONAL only" in SCENARIO_SYSTEM_INSTRUCTION

    def test_db_persist_failure_returns_500(self):
        """Scenario card DB insert returns no data → 500."""
        sb_mock = _make_supabase_mock(persist_ok=False)
        llm_mock = _make_llm_mock(json.dumps(VALID_LLM_JSON))

        with (
            patch("ai.ai_router._get_supabase", return_value=sb_mock),
            patch("ai.ai_router._get_client", return_value=llm_mock),
        ):
            resp = client.post(
                "/ai/assistant/generate-scenario",
                json={"patient_id": PATIENT_ID},
            )
        assert resp.status_code == 500
