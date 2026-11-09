"""Seed ARISE 2026 participants. Idempotent: existing participant_ids are skipped.
Prints each participant's credentials once (they are NOT shown again later).

Email is the login identity — duplicates are skipped.

Usage:
  /arise/venv/bin/python /arise/scripts/seed_participants.py                 # demo names+emails
  /arise/venv/bin/python /arise/scripts/seed_participants.py people.csv      # one 'full name,email' per line
  /arise/venv/bin/python /arise/scripts/seed_participants.py --start 101 people.csv
"""
import secrets
import sys

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
from database import Base, SessionLocal, engine  # noqa: E402
from models import Participant  # noqa: E402

Base.metadata.create_all(bind=engine)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
start_id = 1
if "--start" in sys.argv:
    start_id = int(sys.argv[sys.argv.index("--start") + 1])

if args:
    rows = []
    with open(args[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                name, email = [x.strip() for x in line.split(",", 1)]
            else:
                name, email = line, ""
            rows.append((name, email))
else:
    rows = [
        ("Alya Putri Ramadhani", "alya@example.com"),
        ("Bima Satria Nugraha", "bima@example.com"),
        ("Citra Ayu Lestari", "citra@example.com"),
        ("Dimas Aditya Pratama", "dimas@example.com"),
        ("Eka Wulan Sari", "eka@example.com"),
    ]

db = SessionLocal()
existing_ids = {p.participant_id for p in db.query(Participant).all()}
existing_emails = {p.email.lower() for p in db.query(Participant).all() if p.email}
n = start_id
for name, email in rows:
    email = email.strip().lower()
    if not email:
        print(f"SKIP  {name} | missing email (email is the login identity)")
        continue
    if email in existing_emails:
        print(f"SKIP  {email} | already registered")
        continue
    while f"ARISE{n:03d}" in existing_ids:
        n += 1
    pid = f"ARISE{n:03d}"
    token = secrets.token_hex(8)
    db.add(Participant(participant_id=pid, full_name=name,
                       email=email, token=token))
    db.commit()
    existing_ids.add(pid)
    existing_emails.add(email)
    print(f"CREATED {pid} | {name} | {email} | token={token}")
    n += 1
print("Done.")
