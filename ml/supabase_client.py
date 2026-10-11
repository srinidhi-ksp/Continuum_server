"""
ml/supabase_client.py

Server-side Supabase client for the ML service. Uses the SERVICE ROLE
key, never the anon key the React frontend uses, so it can write
coordination_metric rows regardless of Row Level Security policies.

NEVER expose SUPABASE_SERVICE_ROLE_KEY to the frontend, put it in a
VITE_ variable, or commit it to git. It has full read/write access to
every table.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # safe to call even if main.py already did — idempotent

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

_client = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    from supabase import create_client

    _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
else:
    print(
        "[ml.supabase_client] SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in .env — "
        "/ml/predict will still return predictions, it just won't save them "
        "to coordination_metric. Fine for testing the model, not for the real app."
    )


def get_supabase():
    """Returns the server-side Supabase client, or None if not configured."""
    return _client
