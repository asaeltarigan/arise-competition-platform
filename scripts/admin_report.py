"""ARISE 2026 — organizer admin report.
Lists all participants and all submissions (accepted + rejected).
Usage: /arise/venv/bin/python /arise/scripts/admin_report.py
"""
import sys
from datetime import datetime, timezone

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from database import SessionLocal  # noqa: E402
from models import Participant, Submission  # noqa: E402
from sqlalchemy import func  # noqa: E402

db = SessionLocal()

print("=" * 78)
print("PARTICIPANTS")
print("=" * 78)
for p in db.query(Participant).order_by(Participant.participant_id).all():
    today = datetime.now(timezone.utc).date()
    used_today = db.query(Submission).filter(
        Submission.participant_id == p.participant_id,
        Submission.status == "accepted",
        func.date(Submission.submitted_at) == today.isoformat()).count()
    total = db.query(Submission).filter(
        Submission.participant_id == p.participant_id,
        Submission.status == "accepted").count()
    bests = {}
    for s in db.query(Submission).filter(
            Submission.participant_id == p.participant_id,
            Submission.status == "accepted"):
        cur = bests.get(s.problem)
        if cur is None or s.score > cur:
            bests[s.problem] = s.score
    bstr = " ".join(f"{k}={v:.3f}" for k, v in sorted(bests.items()))
    print(f"  {p.participant_id}  {p.full_name:<28} today={used_today}/5 total={total} best[{bstr}] token={p.token}")

print()
print("=" * 78)
print("SUBMISSIONS")
print("=" * 78)
rows = db.query(Submission).order_by(Submission.submitted_at.desc()).all()
if not rows:
    print("  (none yet)")
for s in rows:
    score = f"{s.score:.4f}" if s.score is not None else "-"
    err = f"  [rejected: {s.error}]" if s.status == "rejected" else ""
    print(f"  {s.submitted_at}  {s.participant_id:<10} {s.problem:<10} {s.status:<8} score={score:<8} rows={s.rows}  {s.filename}{err}")

print()
accepted = db.query(Submission).filter(Submission.status == "accepted").count()
rejected = db.query(Submission).filter(Submission.status == "rejected").count()
print(f"Totals: {accepted} accepted, {rejected} rejected, {accepted + rejected} total.")
