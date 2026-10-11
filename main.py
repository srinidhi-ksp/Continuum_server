"""
Continuum backend — FastAPI entry point.

What this file does:
  1. Loads .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, FRONTEND_ORIGINS)
     BEFORE importing anything from ml/, since ml.supabase_client reads
     those env vars at import time to build its Supabase client.
  2. Creates the FastAPI app and enables CORS so the Vite dev server
     (and later, your deployed frontend) is allowed to call this API —
     without this, the browser blocks the requests even though curl/
     Swagger UI would work fine.
  3. Mounts the ML router, so POST /ml/predict and GET /ml/health become
     real endpoints on this app.

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive testing.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # must run before `from ml...` imports below

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.ml_router import router as ml_router
from ai.ai_router import router as ai_router

app = FastAPI(
    title="Continuum API",
    description="Backend for the Continuum care coordination platform.",
    version="0.1.0",
)

# Comma-separated list in .env, e.g. FRONTEND_ORIGINS=http://localhost:5173
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)
app.include_router(ai_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "continuum-api"}


@app.get("/health")
def health():
    return {"status": "healthy"}
