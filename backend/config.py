"""ARISE 2026 — configuration (multi-problem: Explorer / Innovator / Master)."""
import os
from pathlib import Path

# Deployment root. Defaults to the repository root; override with ARISE_HOME.
BASE_DIR = Path(os.environ.get("ARISE_HOME", Path(__file__).resolve().parent.parent))
BACKEND_DIR = BASE_DIR / "backend"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
LEADERBOARD_DIR = BASE_DIR / "leaderboard"
FRONTEND_DIR = BASE_DIR / "frontend"
PROBLEMS_DIR = BASE_DIR / "problems"

DATABASE_URL = f"sqlite:///{BASE_DIR / 'arise.db'}"
LEADERBOARD_CSV = LEADERBOARD_DIR / "leaderboard.csv"

# Admin access (header X-Admin-Key). Never commit a real key: set ARISE_ADMIN_KEY.
# Empty string disables admin routes rather than exposing a default key.
ADMIN_KEY = os.environ.get("ARISE_ADMIN_KEY", "")
BACKUPS_DIR = BASE_DIR / "backups"

# Competition rules — individual competition
MAX_SUBMISSIONS_PER_DAY = 30  # accepted submissions per participant, per day, ACROSS ALL PROBLEMS (global cap)
ID_COL = "id"
PRED_COL = "prediction"

# The three ARISE 2026 problems (modular — add/remove freely; each folder holds
# train.csv / test.csv / answer.csv under /arise/problems/<key>/).
#   metric: accuracy | f1 | rmse        (rmse is negated so higher = better)
#   answer_label: column name holding the ground truth in answer.csv
PROBLEMS = {
    "explorer": {"name": "Explorer", "weight": 20, "metric": "accuracy", "answer_label": "label"},
    "innovator": {"name": "Innovator", "weight": 30, "metric": "accuracy", "answer_label": "label"},
    "master": {"name": "Master", "weight": 50, "metric": "f1_pos", "answer_label": "label"},
}


def problem_dir(problem: str) -> Path:
    return PROBLEMS_DIR / problem


def problem_test_csv(problem: str) -> Path:
    return problem_dir(problem) / "test.csv"


def problem_answer_csv(problem: str) -> Path:
    return problem_dir(problem) / "answer.csv"


for d in (SUBMISSIONS_DIR, LEADERBOARD_DIR, PROBLEMS_DIR):
    d.mkdir(parents=True, exist_ok=True)
