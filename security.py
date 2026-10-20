import logging
from typing import Callable, Optional

from fastapi import Header, HTTPException

from ml.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def verify_supabase_token(
    authorization: Optional[str],
    supabase_provider: Callable = get_supabase,
) -> Optional[str]:
    if not authorization:
        return None

    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        return None

    supabase = supabase_provider()
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured.")

    try:
        response = supabase.auth.get_user(token.strip())
        user = response.user
        return str(user.id) if user else None
    except Exception as exc:
        logger.info("Supabase bearer-token verification failed: %s", type(exc).__name__)
        return None


def require_caregiver(
    authorization: Optional[str] = Header(default=None),
) -> str:
    caregiver_id = verify_supabase_token(authorization)
    if not caregiver_id:
        raise HTTPException(status_code=401, detail="A valid Supabase bearer token is required.")
    return caregiver_id


def require_active_patient_link(
    caregiver_id: str,
    patient_id: str,
    supabase_provider: Callable = get_supabase,
    primary_only: bool = False,
) -> dict:
    supabase = supabase_provider()
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        result = (
            supabase.from_("caregiver_pat")
            .select("caregiver_id, patient_id, role, status, removed_at, assigned_responsibilities")
            .eq("caregiver_id", caregiver_id)
            .eq("patient_id", patient_id)
            .eq("status", "active")
            .is_("removed_at", None)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.exception("Caregiver-patient membership check failed")
        raise HTTPException(status_code=503, detail="Patient authorization could not be verified.") from exc

    membership = (result.data or [None])[0]
    if not membership:
        raise HTTPException(status_code=403, detail="You are not authorized to access this patient.")
    if primary_only and membership.get("role") != "primary":
        raise HTTPException(status_code=403, detail="Only the primary caregiver can perform this action.")
    return membership


def require_primary_patient_link(
    caregiver_id: str,
    patient_id: str,
    supabase_provider: Callable = get_supabase,
) -> dict:
    return require_active_patient_link(
        caregiver_id,
        patient_id,
        supabase_provider=supabase_provider,
        primary_only=True,
    )
