"""
ml/generate_dataset.py

Generates a synthetic caregiver-week dataset for PROTOTYPE model
development. This is not real data — you must say so explicitly in
your report (see the note at the bottom of this file).

Run:
    python ml/generate_dataset.py

Output:
    ml/data/caregiver_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)   # fixed seed = reproducible dataset
N_ROWS = 500


def clip01(x):
    return np.clip(x, 0, 1)


def normalize(x, lo, hi):
    """Scale x into [0, 1] given an expected realistic range [lo, hi]."""
    return clip01((x - lo) / (hi - lo))


def generate():
    n = N_ROWS

    # --- behavioral features -------------------------------------------------
    tasks_per_day = RNG.poisson(lam=6, size=n) + 2                     # 2–~18
    night_interruptions = RNG.poisson(lam=1.8, size=n)                 # 0–~9
    missed_tasks = RNG.poisson(lam=1.0 + 0.22 * tasks_per_day, size=n) # rises with load
    missed_tasks = np.clip(missed_tasks, 0, tasks_per_day)             # can't miss more than exist

    base_sleep = RNG.normal(loc=7.2, scale=0.9, size=n)
    avg_sleep_hours = np.clip(base_sleep - 0.4 * night_interruptions, 2.5, 9.5)

    late_logs = RNG.poisson(lam=0.5 + 0.12 * tasks_per_day, size=n)
    task_share = clip01(RNG.beta(a=2.2, b=2.0, size=n))                # 0–1, how much of the load falls on this caregiver
    notifications_received = RNG.poisson(lam=3 + 0.5 * tasks_per_day, size=n)
    scenario_attempts = RNG.poisson(lam=1.2, size=n)

    caregiver_week_id = [f"CGW{100000 + i}" for i in range(n)]

    # --- ground-truth strain index (used ONLY to generate labels) -----------
    # weighted combination of normalized risk-increasing features + noise
    # (tighter lo/hi bounds than the realistic max, on purpose, so the score
    #  actually reaches the extremes instead of hugging the middle)
    strain_index = (
        0.20 * normalize(tasks_per_day, 2, 12) +
        0.22 * normalize(missed_tasks, 0, 5) +
        0.18 * normalize(night_interruptions, 0, 6) +
        0.16 * normalize(8 - avg_sleep_hours, 0, 4) +      # sleep deficit
        0.12 * normalize(late_logs, 0, 4) +
        0.08 * task_share +
        0.04 * normalize(notifications_received, 0, 10)
    ) * 100

    noise = RNG.normal(loc=0, scale=4, size=n)
    strain_index = np.clip(strain_index + noise, 0, 100)

    # Load level buckets (Random Forest target)
    load_level = np.where(
        strain_index < 35, "Low",
        np.where(strain_index < 65, "Medium", "High")
    )

    # Care Coordination Score (0-100, HIGHER = healthier/more manageable)
    coordination_score = np.clip(100 - strain_index + RNG.normal(0, 3, n), 0, 100).round(1)

    df = pd.DataFrame({
        "caregiver_week_id": caregiver_week_id,
        "tasks_per_day": tasks_per_day,
        "missed_tasks": missed_tasks,
        "night_interruptions": night_interruptions,
        "avg_sleep_hours": avg_sleep_hours.round(1),
        "late_logs": late_logs,
        "task_share": task_share.round(2),
        "notifications_received": notifications_received,
        "scenario_attempts": scenario_attempts,
        "strain_score": strain_index.round(1),        # keep for reference/report
        "coordination_score": coordination_score,      # what you display 0-100
        "load_level": load_level,                       # Random Forest target
    })

    return df


if __name__ == "__main__":
    df = generate()
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "caregiver_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["load_level"].value_counts())
    print(df.head())

# -----------------------------------------------------------------------
# ACADEMIC HONESTY NOTE (put a version of this in your report):
# "The prototype dataset consists of synthetic caregiver-week records
#  generated within realistic behavioral ranges, because the deployed
#  application does not yet have sufficient longitudinal real usage
#  data. Once real questionnaire responses and live application usage
#  data (DailyCareLog, Notification, Timeline) are available, they will
#  supplement and eventually replace the synthetic set."
# -----------------------------------------------------------------------
