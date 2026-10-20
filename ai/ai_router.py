import os
import json
import uuid
import concurrent.futures
import logging
import time
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional
from security import require_active_patient_link, verify_supabase_token

router = APIRouter(prefix="/ai", tags=["AI Caregiver Assistant"])
logger = logging.getLogger(__name__)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GEMINI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# ── Gemini client (lazy init) ────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    _client = genai.Client(api_key=key)
    return _client


# ── Supabase client (reuse ml's singleton) ───────────────────────────────
def _get_supabase():
    from ml.supabase_client import get_supabase
    return get_supabase()


# ── Auth helpers ─────────────────────────────────────────────────────────
def _verify_token(authorization: Optional[str]) -> Optional[str]:
    """Validate Supabase JWT and return caregiver_id (user.id), or None if no token."""
    return verify_supabase_token(authorization, _get_supabase)


def _require_token(authorization: Optional[str] = Header(default=None)) -> str:
    """Dependency: require a valid Supabase JWT. Returns caregiver_id."""
    caregiver_id = _verify_token(authorization)
    if not caregiver_id:
        raise HTTPException(
            status_code=401,
            detail="A valid Supabase bearer token is required.",
        )
    return caregiver_id


def _require_patient_access(caregiver_id: str, patient_id: str) -> None:
    """Verify the caregiver has an active link to this patient. Raises 403 if not."""
    require_active_patient_link(caregiver_id, patient_id, _get_supabase)


# ── Request / Response schemas ───────────────────────────────────────────
class AIChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="The caregiver's message/question")
    patient_id: Optional[str] = Field(default=None, description="Authorized patient context to use")
    context: dict = Field(default_factory=dict, description="Deprecated; never trusted by the server")
    language: str = Field(default="en", description="Target language code (en | hi | ta)")


class AIChatResponse(BaseModel):
    response: str


class GenerateScenarioRequest(BaseModel):
    patient_id: str = Field(..., description="UUID of the patient to generate a scenario for")


class ScenarioOptionOut(BaseModel):
    option_id: str
    scenario_id: str
    option_text: str
    is_correct: bool
    explanation: str


class ScenarioCardOut(BaseModel):
    scenario_id: str
    patient_id: str
    title: str
    scenario_text: str
    difficulty: str
    category: str
    is_ai_generated: bool


class GenerateScenarioResponse(BaseModel):
    card: ScenarioCardOut
    options: List[ScenarioOptionOut]


# ── Language mapping ─────────────────────────────────────────────────────
LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

# ── Safety system prompt (SRS 3.6 / 3.15) ───────────────────────────────
SCENARIO_SYSTEM_INSTRUCTION = """You are the Continuum Micro-Learning Scenario Generator.

ROLE:
- You create realistic, educational caregiving scenario questions for family caregivers.
- Scenarios help caregivers practice good judgment for everyday care situations.

ABSOLUTE SAFETY RULES (SRS 3.6 / 3.15):
- NEVER provide medical diagnoses or suggest that a patient has any condition beyond what is given.
- NEVER give specific medication dosing instructions, prescribe medication changes, or recommend when to skip/alter medication schedules.
- NEVER replace professional medical advice. If a scenario topic requires clinical judgment, frame it as "consult a healthcare provider."
- Scenarios must be SUPPORTIVE and EDUCATIONAL only — focused on caregiving responses, communication, comfort, safety, and daily care.
- Do NOT include specific drug names unless they are already in the provided patient context.
- Do NOT invent medical facts. Base scenarios only on the patient context provided.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences, matching exactly this shape:
{
  "title": "<short scenario title, max 60 chars>",
  "scenario_text": "<2-3 sentence realistic caregiving situation>",
  "difficulty": "<easy|medium|hard>",
  "category": "<dementia_care|parkinson_care|medication_safety|emotional_support|fall_prevention>",
  "options": [
    {
      "option_text": "<response option A>",
      "is_correct": true,
      "explanation": "<why this is the best caregiving response>"
    },
    {
      "option_text": "<response option B>",
      "is_correct": false,
      "explanation": "<why this is not ideal>"
    },
    {
      "option_text": "<response option C>",
      "is_correct": false,
      "explanation": "<why this is not ideal>"
    }
  ]
}

Rules for options:
- Include 2-4 options total; exactly 1 must have is_correct=true.
- All options must have non-empty explanation strings.
- Keep option_text concise (max 100 chars each).
- The scenario must NOT ask the caregiver to make clinical/medical decisions.
"""


def _call_gemini_with_timeout(client, model, contents, config, timeout_seconds=45):
    """Run Gemini generate_content in a thread with a timeout to avoid hanging."""
    future = _GEMINI_EXECUTOR.submit(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=config,
    )
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Gemini API call timed out after {timeout_seconds}s") from exc


def _is_transient_gemini_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (408, 425, 429, 500, 502, 503, 504):
        return True
    message = str(error).lower()
    return any(
        term in message
        for term in (
            "timeout", "temporarily unavailable", "connection reset",
            "resource_exhausted", "429", "503", "500", "502", "504",
        )
    )


def _generate_with_retry(client, model, contents, config, timeout_seconds=25):
    for attempt in range(2):
        try:
            return _call_gemini_with_timeout(client, model, contents, config, timeout_seconds)
        except Exception as exc:
            if attempt == 0 and _is_transient_gemini_error(exc):
                time.sleep(0.5)
                continue
            raise


@router.post("/chat", response_model=AIChatResponse)
def ai_chat(
    payload: AIChatRequest,
    caregiver_id: str = Depends(_require_token),
):
    """Return guidance using only patient data verified for the authenticated caregiver."""

    # Input validation
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    client = _get_client()
    if not client:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured on the server.")

    prompt_parts = []
    if payload.patient_id:
        _require_patient_access(caregiver_id, payload.patient_id)
        supabase = _get_supabase()
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase is not configured.")
        try:
            patient_result = (
                supabase.from_("patient")
                .select("full_name, condition_type")
                .eq("patient_id", payload.patient_id)
                .limit(1)
                .execute()
            )
            if not patient_result.data:
                raise HTTPException(status_code=404, detail="Patient not found.")
            patient = patient_result.data[0]
            prompt_parts.append(
                f"Patient: {patient.get('full_name', 'Patient')}; "
                f"care condition: {patient.get('condition_type') or 'not recorded'}."
            )
            logs_result = (
                supabase.from_("daily_care_log")
                .select("type, quantity, symptoms, notes, created_at")
                .eq("patient_id", payload.patient_id)
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            for log in logs_result.data or []:
                summary = ", ".join(
                    f"{label}: {log[field]}"
                    for field, label in (("type", "type"), ("quantity", "details"), ("symptoms", "symptoms"), ("notes", "note"))
                    if log.get(field)
                )
                if summary:
                    prompt_parts.append(f"Recent care log ({log.get('created_at', '')}): {summary}")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Authorized assistant context could not be loaded")
            raise HTTPException(status_code=503, detail="Patient context is temporarily unavailable.") from exc

    # ── Build system instruction ─────────────────────────────────────────
    lang_name = LANG_NAMES.get(payload.language, "English")

    system_instruction = f"""You are the Continuum Care Coordination Assistant.

ROLE:
- You provide supportive, empathetic, organized guidance for family caregivers.
- You help caregivers understand care tasks, coordination scores, workload distribution, and medication schedules.

STRICT RULES:
- NEVER diagnose diseases or make medical decisions.
- NEVER invent patient data, medication instructions, or coordination scores.
- If a question requires medical expertise, direct the caregiver to a healthcare professional or emergency service.
- Clearly distinguish between stored patient data and your AI-generated guidance.
- Do NOT expose database IDs, raw JSON, or internal implementation details.

RESPONSE FORMAT:
- Respond in {lang_name}.
- Keep responses concise, well-structured, and actionable.
- Use bullet points or numbered lists for clarity when appropriate.
- Address the caregiver warmly and supportively.
- Format your response with proper paragraphs, headings where useful, and clear spacing.
- Do NOT use raw Markdown symbols like ** or ## — use plain text with natural structure.
"""

    # ── Build prompt with context ────────────────────────────────────────
    context_block = "\n".join(prompt_parts) if prompt_parts else "No patient context requested."

    prompt = f"""=== CARE CONTEXT ===
{context_block}
====================

Caregiver's question: {message}"""

    try:
        from google import genai as _genai
        response = _generate_with_retry(
            client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
            timeout_seconds=25,
        )
        if not getattr(response, "text", None):
            raise HTTPException(status_code=502, detail="The AI returned an empty response. Please try again.")
        return AIChatResponse(response=response.text)
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The AI assistant took too long to respond. Please try again.",
        )
    except Exception as e:
        err_str = str(e)
        logger.exception("AI assistant request failed")
        # Surface actionable errors
        if "API_KEY" in err_str or "credentials" in err_str.lower():
            raise HTTPException(
                status_code=503,
                detail="AI service configuration error. Contact your administrator.",
            )
        if "quota" in err_str.lower() or "rate" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail="AI service rate limit reached. Please wait a moment and try again.",
            )
        if "model" in err_str.lower() and any(word in err_str.lower() for word in ("not found", "unsupported", "not available")):
            raise HTTPException(
                status_code=503,
                detail=f"Configured Gemini model '{GEMINI_MODEL}' is unavailable. Set GEMINI_MODEL to an available model.",
            )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate AI response. Please try again.",
        )


# ── POST /ai/assistant/generate-scenario ─────────────────────────────────
@router.post("/assistant/generate-scenario", response_model=GenerateScenarioResponse)
def generate_scenario(
    payload: GenerateScenarioRequest,
    caregiver_id: str = Depends(_require_token),
):
    """
    Generate a patient-specific AI scenario card + options and persist them
    as real scenario_card / scenario_option rows so scenario_attempt tracking
    works unmodified.

    Safety: system prompt enforces SRS 3.6 / 3.15 — no diagnosis, no
    medication instructions, educational/supportive only.

    Authorization: requires a valid Supabase JWT and an active caregiver_pat
    link to the requested patient_id.
    """
    supabase = _get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase is not configured on this server. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env."
            ),
        )

    # ── 0. Authorization: verify caregiver has access to this patient ─────
    _require_patient_access(caregiver_id, payload.patient_id)

    # ── 1. Fetch patient row ─────────────────────────────────────────────
    patient_res = (
        supabase
        .from_("patient")
        .select("full_name, condition_type, medical_notes")
        .eq("patient_id", payload.patient_id)
        .single()
        .execute()
    )
    if not patient_res.data:
        raise HTTPException(status_code=404, detail="Patient not found.")

    patient = patient_res.data
    patient_name = patient.get("full_name", "the patient")
    condition = patient.get("condition_type", "Unknown condition")
    notes = patient.get("medical_notes") or "No additional notes."

    # ── 2. Fetch recent timeline events (last 5) ─────────────────────────
    timeline_res = (
        supabase
        .from_("time_event")
        .select("title, event_type, created_at")
        .eq("patient_id", payload.patient_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    timeline_items = timeline_res.data or []

    # ── 3. Fetch recent daily care logs (last 5) ─────────────────────────
    logs_res = (
        supabase
        .from_("daily_care_log")
        .select("type, notes, created_at")
        .eq("patient_id", payload.patient_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    log_items = logs_res.data or []

    # ── 4. Build LLM prompt ──────────────────────────────────────────────
    timeline_text = "\n".join(
        f"  - [{item.get('event_type', 'event')}] {item.get('title', '')} "
        f"({item.get('created_at', '')[:10]})"
        for item in timeline_items
    ) or "  (none recorded)"

    logs_text = "\n".join(
        f"  - [{item.get('type', 'log')}] {item.get('notes', '')} "
        f"({item.get('created_at', '')[:10]})"
        for item in log_items
    ) or "  (none recorded)"

    user_prompt = (
        f"Generate a realistic caregiving scenario for a caregiver looking after:\n\n"
        f"Patient: {patient_name}\n"
        f"Condition: {condition}\n"
        f"Medical notes: {notes}\n\n"
        f"Recent timeline events:\n{timeline_text}\n\n"
        f"Recent daily care log entries:\n{logs_text}\n\n"
        "Create one scenario that a family caregiver might realistically face "
        "when caring for this patient. The scenario must be grounded in the "
        "condition and recent care context above.\n"
        "Return ONLY the JSON object described in your instructions — no extra text."
    )

    # ── 5. Call LLM ──────────────────────────────────────────────────────
    client = _get_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Add it to Continuum_server/.env.",
        )

    try:
        from google import genai as _genai
        llm_response = _generate_with_retry(
            client,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=_genai.types.GenerateContentConfig(
                system_instruction=SCENARIO_SYSTEM_INSTRUCTION,
                temperature=0.7,
            ),
            timeout_seconds=25,
        )
        raw_text = llm_response.text.strip()
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Scenario generation timed out. Please try again.",
        )
    except Exception as e:
        err_str = str(e)
        logger.exception("AI scenario generation failed")
        if "quota" in err_str.lower() or "rate" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail="AI service rate limit reached. Please wait a moment and try again.",
            )
        if "model" in err_str.lower() and any(word in err_str.lower() for word in ("not found", "unsupported", "not available")):
            raise HTTPException(
                status_code=503,
                detail=f"Configured Gemini model '{GEMINI_MODEL}' is unavailable. Set GEMINI_MODEL to an available model.",
            )
        raise HTTPException(
            status_code=500,
            detail=f"Scenario generation failed. Please try again.",
        )

    # ── 6. Parse JSON (strip markdown fences if present) ─────────────────
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        raw_text = raw_text.rsplit("```", 1)[0]

    try:
        scenario_json = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[AI generate-scenario] JSON parse error: {e}\nRaw: {raw_text}")
        raise HTTPException(
            status_code=502,
            detail="The AI returned an unexpected format. Please try again.",
        )

    # ── 7. Validate required fields ──────────────────────────────────────
    if not isinstance(scenario_json, dict):
        raise HTTPException(status_code=502, detail="The AI returned an unexpected scenario format.")
    required_fields = {"title", "scenario_text", "difficulty", "category", "options"}
    missing = required_fields - set(scenario_json.keys())
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"AI response missing required fields: {missing}",
        )

    options_raw = scenario_json["options"]
    if not isinstance(options_raw, list) or not (2 <= len(options_raw) <= 4):
        raise HTTPException(status_code=502, detail="AI must return 2-4 options.")
    if any(not isinstance(option, dict) for option in options_raw):
        raise HTTPException(status_code=502, detail="AI scenario options have an invalid format.")
    if sum(1 for o in options_raw if o.get("is_correct") is True) != 1:
        raise HTTPException(
            status_code=502, detail="AI must mark exactly one option as correct."
        )

    # ── 8. Persist scenario_card row ─────────────────────────────────────
    card_id = str(uuid.uuid4())
    card_row = {
        "scenario_id": card_id,
        "patient_id": payload.patient_id,
        "title": scenario_json["title"][:120],
        "scenario_text": scenario_json["scenario_text"],
        "difficulty": scenario_json["difficulty"],
        "category": scenario_json["category"],
        "is_ai_generated": True,
    }
    card_res = supabase.from_("scenario_card").insert(card_row).execute()
    if not card_res.data:
        raise HTTPException(
            status_code=500, detail="Failed to save scenario card to database."
        )

    # ── 9. Persist scenario_option rows ──────────────────────────────────
    option_rows = [
        {
            "option_id": str(uuid.uuid4()),
            "scenario_id": card_id,
            "option_text": opt.get("option_text", "")[:200],
            "is_correct": bool(opt.get("is_correct", False)),
            "explanation": opt.get("explanation", ""),
        }
        for opt in options_raw
    ]
    opts_res = supabase.from_("scenario_option").insert(option_rows).execute()
    if not opts_res.data:
        raise HTTPException(
            status_code=500, detail="Failed to save scenario options to database."
        )

    # ── 10. Return shaped response ────────────────────────────────────────
    return GenerateScenarioResponse(
        card=ScenarioCardOut(**card_res.data[0]),
        options=[ScenarioOptionOut(**o) for o in opts_res.data],
    )
