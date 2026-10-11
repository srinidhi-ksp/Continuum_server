import os
import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

router = APIRouter(prefix="/ai", tags=["AI Caregiver Assistant"])

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


# ── Request / Response schemas ───────────────────────────────────────────
class AIChatRequest(BaseModel):
    message: str = Field(..., description="The caregiver's message/question")
    context: dict = Field(default_factory=dict, description="Aggregated patient context")
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


@router.post("/chat", response_model=AIChatResponse)
def ai_chat(payload: AIChatRequest):
    """Caregiver AI Assistant — receives context + message, returns guidance."""

    client = _get_client()

    if not client:
        # Graceful offline fallback when no API key is configured
        patient_name = (
            payload.context.get("patientSummary", {}).get("name", "Unknown")
        )
        offline_msg = (
            f"I'm running in offline mode (no GEMINI_API_KEY configured). "
            f"I received your question about '{payload.message}' "
            f"with context for patient {patient_name}. "
            f"Add GEMINI_API_KEY to Continuum_server/.env and restart the server."
        )
        return AIChatResponse(response=offline_msg)

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
"""

    # ── Build prompt with context ────────────────────────────────────────
    ctx = payload.context
    prompt_parts = []

    patient_summary = ctx.get("patientSummary")
    if patient_summary:
        prompt_parts.append(
            f"Patient: {patient_summary.get('name', 'Unknown')}, "
            f"Age: {patient_summary.get('age', 'N/A')}, "
            f"Condition: {patient_summary.get('condition', 'N/A')}"
        )

    if ctx.get("caregiverName"):
        prompt_parts.append(f"Caregiver asking: {ctx['caregiverName']}")

    coord = ctx.get("coordinationScore") or ctx.get("latestMetric")
    if coord:
        prompt_parts.append(
            f"Care Coordination Score: {coord.get('score', 'N/A')}/100, "
            f"Risk Level: {coord.get('risk_level', 'N/A')}, "
            f"Recommendation: {coord.get('recommendation', 'N/A')}"
        )

    if ctx.get("medsTotal") is not None:
        prompt_parts.append(
            f"Medications: {ctx.get('medsTaken', 0)} of {ctx['medsTotal']} taken today"
        )
    if ctx.get("careCompletion") is not None:
        prompt_parts.append(f"Today's care completion: {ctx['careCompletion']}%")

    caregivers = ctx.get("caregivers", [])
    if caregivers:
        active = sum(1 for c in caregivers if c.get("active"))
        prompt_parts.append(f"Care circle: {len(caregivers)} members ({active} active)")

    context_block = "\n".join(prompt_parts) if prompt_parts else "No patient context provided."

    prompt = f"""=== CARE CONTEXT ===
{context_block}
====================

Caregiver's question: {payload.message}"""

    try:
        from google import genai as _genai
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=_genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return AIChatResponse(response=response.text)
    except Exception as e:
        print(f"[AI Error] {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate AI response: {str(e)}",
        )


# ── POST /ai/assistant/generate-scenario ─────────────────────────────────
@router.post("/assistant/generate-scenario", response_model=GenerateScenarioResponse)
def generate_scenario(payload: GenerateScenarioRequest):
    """
    Generate a patient-specific AI scenario card + options and persist them
    as real scenario_card / scenario_option rows so scenario_attempt tracking
    works unmodified.

    Safety: system prompt enforces SRS 3.6 / 3.15 — no diagnosis, no
    medication instructions, educational/supportive only.
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
        llm_response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_prompt,
            config=_genai.types.GenerateContentConfig(
                system_instruction=SCENARIO_SYSTEM_INSTRUCTION,
                temperature=0.7,
            ),
        )
        raw_text = llm_response.text.strip()
    except Exception as e:
        print(f"[AI generate-scenario Error] {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}",
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
    if sum(1 for o in options_raw if o.get("is_correct")) != 1:
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
        "difficulty": scenario_json.get("difficulty", "medium"),
        "category": scenario_json.get("category", "dementia_care"),
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
