"""ARISE 2026 — SAFE reset: backs up everything, then wipes demo data.
Use this instead of raw `rm` commands. The backup is created automatically,
so nothing is ever lost without a copy.

Usage: /arise/venv/bin/python /arise/scripts/reset_db.py
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import BASE_DIR, BACKUPS_DIR, LEADERBOARD_CSV  # noqa: E402

# 1) Backup first (same layout as backup.py).
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BACKUPS_DIR / f"pre_reset_{stamp}"
out.mkdir(parents=True, exist_ok=True)
for src, name in [(BASE_DIR / "arise.db", "arise.db"),
                  (BASE_DIR / "submissions", "submissions"),
                  (LEADERBOARD_CSV, "leaderboard.csv")]:
    if src.is_dir():
        shutil.copytree(src, out / name)
    elif src.exists():
        shutil.copy2(src, out / name)
print(f"Backup saved to {out}")

# 2) Reset (DB + submissions + leaderboard export).
for p in (BASE_DIR / "arise.db", LEADERBOARD_CSV):
    if p.exists():
        p.unlink()
subs = BASE_DIR / "submissions"
if subs.exists():
    shutil.rmtree(subs)
subs.mkdir(exist_ok=True)
print("Reset complete. Restarting the service to recreate the DB...")

import subprocess  # noqa: E402
subprocess.run(["sudo", "systemctl", "restart", "arise"], check=False)
print("Done. Re-seed participants with: seed_participants.py")
