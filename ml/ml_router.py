"""
ml/ml_router.py

FastAPI router exposing POST /ml/predict.
Mount this into your existing FastAPI app (see bottom of file for how).

Run standalone for testing:
    uvicorn ml.ml_router:app --reload
Then open:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from security import require_primary_patient_link

from ml.supabase_client import get_supabase

MODELS_DIR = Path(__file__).parent / "models"

FEATURES = [
    "tasks_per_day",
    "missed_tasks",
    "night_interruptions",
    "avg_sleep_hours",
    "late_logs",
    "task_share",
    "notifications_received",
    "scenario_attempts",
]

# Anchor scores used to turn RF class probabilities into a 0-100 score.
# Tune these once you have real feedback on what "feels right".
LOAD_LEVEL_ANCHORS = {"Low": 90, "Medium": 60, "High": 25}


# --- request/response schemas -----------------------------------------------
class CaregiverWeekFeatures(BaseModel):
    tasks_per_day: float = Field(..., ge=0, le=30, description="Avg care tasks per day")
    missed_tasks: float = Field(..., ge=0, le=30, description="Tasks missed this week / day")
    night_interruptions: float = Field(..., ge=0, le=15)
    avg_sleep_hours: float = Field(..., ge=0, le=14)
    late_logs: float = Field(..., ge=0, le=20)
    task_share: float = Field(..., ge=0, le=1, description="Fraction of total caregiving this person handles")
    notifications_received: float = Field(..., ge=0, le=50)
    scenario_attempts: float = Field(default=0, ge=0, le=50)

    # Optional: when both are provided, the prediction is also written to
    # coordination_metric in Supabase (via the service-role client) so the
    # React dashboard picks it up on its next read, with no extra frontend
    # write needed. Omit both to just try the model without touching the DB.
    patient_id: Optional[str] = Field(None, description="UUID of the patient this prediction is for")
    caregiver_id: Optional[str] = Field(None, description="UUID of the caregiver submitting this week's data")

    class Config:
        json_schema_extra = {
            "example": {
                "tasks_per_day": 9,
                "missed_tasks": 3,
                "night_interruptions": 4,
                "avg_sleep_hours": 5.2,
                "late_logs": 2,
                "task_share": 0.78,
                "notifications_received": 7,
                "scenario_attempts": 1,
                "patient_id": "22222222-2222-2222-2222-222222222201",
                "caregiver_id": "11111111-1111-1111-1111-111111111101",
            }
        }


class PredictionResponse(BaseModel):
    load_level: Literal["Low", "Medium", "High"]
    load_level_probabilities: dict
    is_anomaly: bool
    anomaly_score: float
    coordination_score: float
    risk_level: Literal["low", "moderate", "elevated"]
    recommendation: str
    saved: bool = Field(..., description="True if this prediction was written to coordination_metric")
    metric_id: Optional[str] = Field(None, description="The new coordination_metric row's id, if saved")


# --- model loading (once, at import time) -----------------------------------
def _load_models():
    load_path = MODELS_DIR / "caregiver_load_model.joblib"
    anomaly_path = MODELS_DIR / "caregiver_anomaly_model.joblib"
    if not load_path.exists() or not anomaly_path.exists():
        raise FileNotFoundError(
            "Trained models not found. Run `python ml/generate_dataset.py` "
            "then `python ml/train_models.py` first."
        )
    return joblib.load(load_path), joblib.load(anomaly_path)


rf_model, iso_model = _load_models()

router = APIRouter(prefix="/ml", tags=["ml"])
bearer_auth = HTTPBearer(auto_error=False)


def _authenticated_caregiver_id(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="A Supabase bearer token is required")

    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")

    try:
        user_response = supabase.auth.get_user(credentials.credentials)
        user = user_response.user
        caregiver_id = user.id if user else None
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token") from exc

    if not caregiver_id:
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase token")
    return str(caregiver_id)


def _to_row(features: CaregiverWeekFeatures) -> pd.DataFrame:
    # A DataFrame with the training-time column names, not a bare numpy
    # array — avoids sklearn's "X does not have valid feature names"
    # warning on every single prediction, since both models were fit on
    # a DataFrame in train_models.py.
    values = [0 if feature == "scenario_attempts" else getattr(features, feature) for feature in FEATURES]
    return pd.DataFrame([values], columns=FEATURES)


def _coordination_score(probabilities: dict, is_anomaly: bool) -> float:
    score = sum(probabilities[label] * anchor for label, anchor in LOAD_LEVEL_ANCHORS.items())
    if is_anomaly:
        score -= 12   # anomalies pull the score down regardless of predicted class
    return float(np.clip(score, 0, 100))


def _risk_level(score: float) -> str:
    if score >= 80:
        return "low"
    if score >= 45:
        return "moderate"
    return "elevated"


def _recommendation(risk_level: str, is_anomaly: bool) -> str:
    if is_anomaly:
        return (
            "This week's pattern is noticeably different from usual. "
            "Consider checking in directly and reviewing recent logs together."
        )
    if risk_level == "elevated":
        return "Workload is elevated — consider redistributing tasks among caregivers."
    if risk_level == "moderate":
        return "Care coordination looks manageable, keep an eye on missed tasks and sleep."
    return "Care coordination looks healthy this week."


@router.post("/predict", response_model=PredictionResponse)
def predict(
    features: CaregiverWeekFeatures,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
):
    authenticated_caregiver_id = _authenticated_caregiver_id(credentials)
    if features.caregiver_id and features.caregiver_id != authenticated_caregiver_id:
        raise HTTPException(status_code=403, detail="caregiver_id must match the authenticated user")

    patient_id = features.patient_id
    if patient_id:
        require_primary_patient_link(authenticated_caregiver_id, patient_id)

    try:
        X = _to_row(features)

        proba = rf_model.predict_proba(X)[0]
        probabilities = dict(zip(rf_model.classes_, proba.round(3).tolist()))
        load_level = rf_model.classes_[np.argmax(proba)]

        anomaly_raw_pred = iso_model.predict(X)[0]          # -1 anomaly, 1 normal
        is_anomaly = bool(anomaly_raw_pred == -1)
        anomaly_score = float(iso_model.decision_function(X)[0])  # lower = more abnormal

        coordination_score = round(_coordination_score(probabilities, is_anomaly), 1)
        risk_level = _risk_level(coordination_score)
        recommendation = _recommendation(risk_level, is_anomaly)

        saved = False
        metric_id = None
        supabase = get_supabase()

        if supabase and patient_id:
            result = (
                supabase.table("coordination_metric")
                .insert(
                    {
                        "patient_id": patient_id,
                        "caregiver_id": authenticated_caregiver_id,
                        "score": coordination_score,
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                    }
                )
                .execute()
            )
            if result.data:
                saved = True
                metric_id = result.data[0].get("metric_id")

        return PredictionResponse(
            load_level=load_level,
            load_level_probabilities=probabilities,
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            coordination_score=coordination_score,
            risk_level=risk_level,
            recommendation=recommendation,
            saved=saved,
            metric_id=metric_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
def ml_health():
    """Quick check that both models loaded and whether Supabase writes are configured."""
    return {
        "status": "ok",
        "models_loaded": rf_model is not None and iso_model is not None,
        "rf_classes": list(rf_model.classes_),
        "supabase_configured": get_supabase() is not None,
    }


# --- standalone runnable app (for `uvicorn ml.ml_router:app --reload`) ------
app = FastAPI(title="CONTINUUM ML Service (standalone test app)")
app.include_router(router)

# -----------------------------------------------------------------------
# TO MOUNT INTO YOUR EXISTING FASTAPI APP (app/main.py), do this instead
# of running this file standalone:
#
#   from ml.ml_router import router as ml_router
#   app.include_router(ml_router)
#
# Then your real endpoint becomes: POST /ml/predict on your main app,
# alongside all your other routes.
# -----------------------------------------------------------------------
