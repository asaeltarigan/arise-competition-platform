"""ARISE 2026 — backup the database + submissions + leaderboard.
Run manually or via cron (daily). Keeps the 14 most recent backups.
Usage: /arise/venv/bin/python /arise/scripts/backup.py
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from config import BACKUPS_DIR, BASE_DIR, LEADERBOARD_CSV  # noqa: E402

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BACKUPS_DIR / stamp
out.mkdir(parents=True, exist_ok=True)

sources = [
    (BASE_DIR / "arise.db", "arise.db"),
    (BASE_DIR / "submissions", "submissions"),
    (LEADERBOARD_CSV, "leaderboard.csv"),
]
for src, name in sources:
    if src.is_dir():
        shutil.copytree(src, out / name)
    elif src.exists():
        shutil.copy2(src, out / name)
    else:
        print(f"  (skip) {src} not found")

# Prune: keep the 14 most recent backups.
all_backups = sorted(BACKUPS_DIR.glob("20*"))
for old in all_backups[:-14]:
    shutil.rmtree(old, ignore_errors=True)

print(f"Backup created: {out}")
print(f"Total backups kept: {min(len(all_backups), 14)}")
