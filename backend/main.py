"""ARISE 2026 — FastAPI submission portal backend (individual, multi-problem).

Problems: explorer / innovator / master — each with its own train/test/answer
datasets and metric under /arise/problems/<key>/. All share one pipeline:
upload → validate → score → leaderboard.

Run:  /arise/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
"""
import csv
import hashlib
import hmac
import io
import re
import secrets
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import (
    ADMIN_KEY,
    BACKUPS_DIR,
    BASE_DIR,
    FRONTEND_DIR,
    LEADERBOARD_CSV,
    MAX_SUBMISSIONS_PER_DAY,
    PROBLEMS,
    PROBLEMS_DIR,
    SUBMISSIONS_DIR,
    problem_answer_csv,
    problem_test_csv,
)
from database import Base, engine, get_db
from evaluator import evaluate
from models import Participant, Submission
from validator import ValidationError, validate_submission

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ARISE 2026 Submission System")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── leaderboard reveal policy ─────────────────────────────────
# Public board shows only the top N until the deadline passes,
# then the FULL board is revealed (the race hides, the finish reveals).
CLOSE_DEADLINE_UTC = datetime(2026, 8, 28, 16, 59, 0, tzinfo=timezone.utc).replace(tzinfo=None)  # 23:59 WIB
FINALIST_CUT = 10


def _deadline_revealed() -> bool:
    return _utcnow() >= CLOSE_DEADLINE_UTC


def _neighborhood(rows, participant_id, window=2):
    """Rows around a participant (±window), plus their index (or None)."""
    for i, r in enumerate(rows):
        if r["participant_id"] == participant_id:
            lo, hi = max(0, i - window), min(len(rows), i + window + 1)
            return rows[lo:hi], i
    return [], None


def _cut_info(rows, participant_id):
    """Top-10 cut info for a participant on a board."""
    for i, r in enumerate(rows):
        if r["participant_id"] == participant_id:
            rank = i + 1
            if rank <= FINALIST_CUT:
                below = rows[FINALIST_CUT] if len(rows) > FINALIST_CUT else None
                return {
                    "in_top10": True,
                    "rank": rank,
                    "points_ahead_of_11th": round(r["score"] - below["score"], 4) if below else None,
                }
            cut = rows[FINALIST_CUT - 1]
            return {
                "in_top10": False,
                "rank": rank,
                "points_behind_top10": round(cut["score"] - r["score"], 4),
            }
    return None


def _get_participant_or_401(db: Session, email: str, token: str) -> Participant:
    email = email.strip().lower()
    p = (
        db.query(Participant)
        .filter(func.lower(Participant.email) == email)
        .first()
    )
    if p is None or not hmac.compare_digest(p.token, token):
        raise HTTPException(status_code=401, detail="Invalid email or token")
    return p


def _check_problem(problem: str) -> None:
    if problem not in PROBLEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown problem '{problem}'. Choose from: {', '.join(PROBLEMS)}.",
        )


def _accepted_today(db: Session, participant_id: str) -> int:
    """Accepted submissions today, counted ACROSS ALL problems (global daily cap)."""
    today = _utcnow().date()
    return (
        db.query(func.count(Submission.id))
        .filter(
            Submission.participant_id == participant_id,
            Submission.status == "accepted",
            func.date(Submission.submitted_at) == today.isoformat(),
        )
        .scalar()
        or 0
    )


def _best_scores(db: Session):
    """Best score per (participant, problem) + the earliest time that best score was achieved.

    Tie-break rule (Kaggle-style): equal best scores rank by who achieved the score first.
    """
    best = {}
    best_at = {}
    for s in db.query(Submission).filter(Submission.status == "accepted").all():
        if s.score is None:
            continue
        key = (s.participant_id, s.problem)
        cur = best.get(key)
        if cur is None or s.score > cur:
            best[key] = s.score
            best_at[key] = s.submitted_at
        elif s.score == cur and s.submitted_at < best_at[key]:
            # same best score submitted earlier -> keep the earlier achievement time
            best_at[key] = s.submitted_at
    return best, best_at


def _rebuild_leaderboard(db: Session, problem: str = None):
    """Rows for one problem, or the overall weighted board (20/30/50).
    Writes leaderboard.csv for the requested view too.
    """
    names = {p.participant_id: p.full_name for p in db.query(Participant).all()}
    best, best_at = _best_scores(db)

    if problem is not None:
        w = PROBLEMS[problem]["weight"]
        entries = {pid: s for (pid, pr), s in best.items() if pr == problem}
        ranked = sorted(
            entries.items(),
            key=lambda kv: (-kv[1], best_at[(kv[0], problem)]),
        )
        rows = [
            {"rank": i + 1, "participant_id": pid, "name": names.get(pid, pid),
             "score": round(s * w, 4), "raw": round(s, 4),
             "max": w, "problem": problem}
            for i, (pid, s) in enumerate(ranked)
        ]
    else:
        # Overall: weighted sum of each participant's best per problem (20/30/50).
        weights = {pid: PROBLEMS[pid]["weight"] for pid in PROBLEMS}
        overall = {}
        problems_of = {}
        for (pid, pr), s in best.items():
            overall[pid] = overall.get(pid, 0.0) + s * weights[pr]
            problems_of.setdefault(pid, {})[pr] = round(s, 4)
        # Tie-break: the moment the participant completed their current set of scores
        # (latest per-problem achievement time) - earlier completion ranks higher.
        ranked = sorted(
            overall.items(),
            key=lambda kv: (-kv[1], max(best_at[(kv[0], pr)] for pr in problems_of[kv[0]])),
        )
        rows = [
            {"rank": i + 1, "participant_id": pid, "name": names.get(pid, pid),
             "score": round(s, 4), "problem": "overall",
             "problems": problems_of.get(pid, {})}
            for i, (pid, s) in enumerate(ranked)
        ]

    with open(LEADERBOARD_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "participant_id", "name", "score", "raw", "problem"])
        w.writeheader()
        csv_rows = [{k: r.get(k, "") for k in ("rank", "participant_id", "name", "score", "raw", "problem")} for r in rows]
        w.writerows(csv_rows)
    return rows


def _rank_of(rows, participant_id):
    for r in rows:
        if r["participant_id"] == participant_id:
            return r["rank"]
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/problems")
def problems_list():
    return [
        {"id": pid, "name": p["name"], "weight": p["weight"], "metric": p["metric"]}
        for pid, p in PROBLEMS.items()
    ]


@app.post("/register")
def register(
    full_name: str = Form(...),
    school: str = Form(""),
    email: str = Form(...),
    db: Session = Depends(get_db),
):
    full_name = " ".join(full_name.strip().split())
    email = email.strip().lower()
    if len(full_name) < 3:
        raise HTTPException(status_code=400, detail="Full name is too short.")
    if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    dup = (
        db.query(Participant)
        .filter(func.lower(Participant.email) == email)
        .first()
    )
    if dup:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{dup.email}' is already registered as {dup.participant_id}. "
                "If this is you, contact the organizer to get your token."
            ),
        )

    # Next free ARISE### id.
    ids = [p.participant_id for p in db.query(Participant).all()]
    nums = [int(pid[5:]) for pid in ids if pid.startswith("ARISE") and pid[5:].isdigit()]
    n = (max(nums) + 1) if nums else 1
    pid = f"ARISE{n:03d}"

    token = secrets.token_hex(8)
    db.add(Participant(participant_id=pid, full_name=full_name,
                       school=school or None, email=email or None, token=token))
    db.commit()

    return {
        "participant_id": pid,
        "full_name": full_name,
        "token": token,
        "note": "Save these credentials now — the token is shown only once and is NOT emailed to you. "
                "Use them to submit your prediction CSV (any filename); if you lose them, use "
                "'Forgot your token?' with your email.",
    }


@app.post("/recover")
def recover(
    email: str = Form(...),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    p = (
        db.query(Participant)
        .filter(func.lower(Participant.email) == email)
        .first()
    )
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"No participant found with the email '{email}'. "
                   "Check the spelling or register first.",
        )
    return {
        "participant_id": p.participant_id,
        "email": p.email,
        "full_name": p.full_name,
        "token": p.token,
    }


@app.get("/api/submissions-open")
def submissions_open():
    return {"open": not (SUBMISSIONS_DIR.parent / "submissions_open.flag").exists()}


@app.post("/api/my-rank")
def my_rank(
    email: str = Form(...),
    token: str = Form(...),
    problem: str = Form("overall"),
    db: Session = Depends(get_db),
):
    """Participant-only view: their rank ±2 neighbors on a board (email + token)."""
    participant = _get_participant_or_401(db, email, token)
    if problem != "overall":
        _check_problem(problem)
    rows = _rebuild_leaderboard(db, None if problem == "overall" else problem)
    nb, i = _neighborhood(rows, participant.participant_id)
    cut = _cut_info(rows, participant.participant_id)
    return {
        "participant_id": participant.participant_id,
        "name": participant.full_name,
        "problem": problem,
        "neighborhood": [
            {
                "rank": r["rank"],
                "participant_id": r["participant_id"],
                "name": r["name"],
                "score": r["score"],
                "you": r["participant_id"] == participant.participant_id,
            }
            for r in nb
        ],
        "cut": cut,
        "revealed": _deadline_revealed(),
        "finalist_cut": FINALIST_CUT,
        "deadline": "2026-08-28T23:59:00+07:00",
    }


@app.post("/submit")
async def submit(
    email: str = Form(...),
    token: str = Form(...),
    file: UploadFile = File(...),
    problem: str = Form("explorer"),
    db: Session = Depends(get_db),
):
    participant = _get_participant_or_401(db, email, token)
    participant_id = participant.participant_id
    _check_problem(problem)

    # Global submissions gate: closed when the flag file exists (create/delete
    # /arise/submissions_open.flag to close/open; no code change or restart of
    # logic needed beyond the service reload below).
    if (SUBMISSIONS_DIR.parent / "submissions_open.flag").exists():
        raise HTTPException(
            status_code=403,
            detail=(
                "Submissions are closed. The competition period has ended - "
                "thank you for participating!"
            ),
        )

    used_today = _accepted_today(db, participant_id)
    if used_today >= MAX_SUBMISSIONS_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily submission limit reached "
                   f"({MAX_SUBMISSIONS_PER_DAY} across all competitions). Try again tomorrow.",
        )

    # Persist raw file first (audit trail), then validate + score.
    stamp = _utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = Path(file.filename or "submission.csv").name
    dest = SUBMISSIONS_DIR / f"{participant_id}_{problem}_{stamp}_{safe_name}"
    content = await file.read()
    dest.write_bytes(content)

    # Reject exact re-uploads (same file content already accepted for this problem).
    file_md5 = hashlib.md5(content).hexdigest()
    dup = (
        db.query(Submission)
        .filter(
            Submission.participant_id == participant_id,
            Submission.problem == problem,
            Submission.status == "accepted",
            Submission.md5 == file_md5,
        )
        .first()
    )
    if dup:
        raise HTTPException(
            status_code=400,
            detail=(
                "Duplicate submission: this exact file was already submitted "
                f"(score {dup.score:.4f}). Please upload a different file."
            ),
        )

    try:
        n_rows = validate_submission(dest, safe_name, problem_test_csv(problem))
        score = evaluate(dest, problem_answer_csv(problem),
                         PROBLEMS[problem]["metric"])
    except ValidationError as e:
        db.add(Submission(participant_id=participant_id, problem=problem,
                          filename=safe_name, md5=file_md5,
                          status="rejected", error=str(e)))
        db.commit()
        raise HTTPException(status_code=400, detail=f"Submission rejected: {e}")
    except Exception as e:  # scoring crash (label-type mixes etc.) — never a plain-text 500
        db.add(Submission(participant_id=participant_id, problem=problem,
                          filename=safe_name, md5=file_md5,
                          status="rejected", error=f"scoring error: {e}"))
        db.commit()
        raise HTTPException(status_code=400,
                            detail=f"Submission could not be scored: {e}")

    db.add(Submission(participant_id=participant_id, problem=problem,
                      filename=safe_name, md5=file_md5,
                      status="accepted", score=score, rows=n_rows))
    db.commit()

    rows = _rebuild_leaderboard(db, problem)
    rank = _rank_of(rows, participant_id)
    remaining = MAX_SUBMISSIONS_PER_DAY - used_today - 1

    # neighborhood on the submitted problem's board + overall top-10 cut info
    nb, _ = _neighborhood(rows, participant_id)
    overall_rows = _rebuild_leaderboard(db, None)
    overall_rank = _rank_of(overall_rows, participant_id)
    cut = _cut_info(overall_rows, participant_id)

    return {
        "status": "accepted",
        "problem": problem,
        "problem_name": PROBLEMS[problem]["name"],
        "score": score,
        "rank": rank,
        "submissions_today": used_today + 1,
        "remaining_today": remaining,
        "neighborhood": [
            {
                "rank": r["rank"],
                "participant_id": r["participant_id"],
                "name": r["name"],
                "score": r["score"],
                "you": r["participant_id"] == participant_id,
            }
            for r in nb
        ],
        "overall_rank": overall_rank,
        "cut": cut,
        "revealed": _deadline_revealed(),
        "finalist_cut": FINALIST_CUT,
    }


@app.get("/leaderboard")
def leaderboard(problem: str = None, db: Session = Depends(get_db)):
    if problem is not None:
        _check_problem(problem)
    rows = _rebuild_leaderboard(db, problem)
    # First Solver: earliest accepted submission per problem.
    first_by = {}
    for s in db.query(Submission).filter(Submission.status == "accepted").order_by(Submission.submitted_at).all():
        if s.problem not in first_by:
            first_by[s.problem] = s.participant_id
    out = []
    for r in rows:
        item = {"rank": r["rank"], "name": r["name"], "score": r["score"],
                "problem": r["problem"]}
        if r.get("problems"):
            item["problems"] = r["problems"]
        if r.get("raw") is not None:
            item["raw"] = r["raw"]
        if r.get("max"):
            item["max"] = r["max"]
        if problem is not None:
            item["first_solver"] = (r["participant_id"] == first_by.get(problem))
        item["finalist"] = r["rank"] <= FINALIST_CUT
        out.append(item)

    revealed = _deadline_revealed()
    public_rows = out if revealed else out[:FINALIST_CUT]
    return {
        "rows": public_rows,
        "revealed": revealed,
        "total": len(rows),
        "finalist_cut": FINALIST_CUT,
        "deadline": "2026-08-28T23:59:00+07:00",
    }


@app.get("/")
def portal():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/leaderboard.html")
def leaderboard_page():
    return FileResponse(str(FRONTEND_DIR / "leaderboard.html"))


@app.get("/register.html")
def register_page():
    return FileResponse(str(FRONTEND_DIR / "register.html"))


@app.get("/admin")
def admin_page():
    return FileResponse(str(FRONTEND_DIR / "admin.html"))


def _check_admin(x_admin_key: str = Header(default="")):
    if not hmac.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _log_admin(action: str, detail: str):
    """Append to the admin audit log (append-only CSV)."""
    try:
        with open(BASE_DIR / "admin_log.csv", "a") as f:
            f.write(f"{_utcnow().isoformat()},{action},{detail}\n")
    except Exception:  # noqa: BLE001 — logging must never break the request
        pass


@app.get("/admin/data")
def admin_data(
    db: Session = Depends(get_db),
    x_admin_key: str = Header(default=""),
):
    """READ-ONLY dump for the organizer. No writes happen here."""
    _check_admin(x_admin_key)

    participants = [
        {
            "participant_id": p.participant_id,
            "full_name": p.full_name,
            "school": p.school,
            "email": p.email,
            "token": p.token,
        }
        for p in db.query(Participant).order_by(Participant.participant_id).all()
    ]
    pname = {p["participant_id"]: p["full_name"] for p in participants}
    submissions = [
        {
            "id": s.id,
            "participant_id": s.participant_id,
            "name": pname.get(s.participant_id, s.participant_id),
            "problem": s.problem,
            "filename": s.filename,
            "status": s.status,
            "score": s.score,
            "rows": s.rows,
            "error": s.error,
            "submitted_at": s.submitted_at.isoformat() if s.submitted_at else None,
        }
        for s in db.query(Submission).order_by(Submission.submitted_at.desc()).all()
    ]
    board = _rebuild_leaderboard(db, None)

    return {
        "participants": participants,
        "submissions": submissions,
        "leaderboard": board,
        "problems": [
            {"id": pid, "name": p["name"], "weight": p["weight"], "metric": p["metric"]}
            for pid, p in PROBLEMS.items()
        ],
        "note": "Read-only view. All writes happen through the submission API only.",
    }


@app.post("/admin/participants")
def admin_add_participant(
    full_name: str = Form(...),
    school: str = Form(""),
    email: str = Form(...),
    db: Session = Depends(get_db),
    x_admin_key: str = Header(default=""),
):
    _check_admin(x_admin_key)
    full_name = " ".join(full_name.strip().split())
    email = email.strip().lower()
    if len(full_name) < 3:
        raise HTTPException(status_code=400, detail="Full name is too short.")
    if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    dup = (
        db.query(Participant)
        .filter(func.lower(Participant.email) == email)
        .first()
    )
    if dup:
        raise HTTPException(
            status_code=400,
            detail=f"'{dup.email}' already exists as {dup.participant_id}.",
        )

    ids = [p.participant_id for p in db.query(Participant).all()]
    nums = [int(pid[5:]) for pid in ids if pid.startswith("ARISE") and pid[5:].isdigit()]
    n = (max(nums) + 1) if nums else 1
    pid = f"ARISE{n:03d}"
    token = secrets.token_hex(8)
    db.add(Participant(participant_id=pid, full_name=full_name,
                       school=school or None, email=email or None, token=token))
    db.commit()
    _log_admin("add_participant", pid)
    return {"participant_id": pid, "full_name": full_name, "token": token}


@app.delete("/admin/participants/{participant_id}")
def admin_delete_participant(
    participant_id: str,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(default=""),
):
    _check_admin(x_admin_key)
    p = db.query(Participant).filter(Participant.participant_id == participant_id).first()
    if p is None:
        raise HTTPException(status_code=404, detail="Participant not found.")
    deleted_subs = db.query(Submission).filter(
        Submission.participant_id == participant_id).delete()
    db.delete(p)
    db.commit()
    _log_admin("delete_participant", f"{participant_id} ({p.full_name}, {deleted_subs} submissions)")
    return {"deleted": participant_id, "submissions_removed": deleted_subs}


@app.delete("/admin/submissions/{submission_id}")
def admin_delete_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(default=""),
):
    _check_admin(x_admin_key)
    s = db.get(Submission, submission_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Submission not found.")
    _log_admin("delete_submission",
               f"{s.participant_id} {s.problem} {s.status} score={s.score}")
    db.delete(s)
    db.commit()
    return {"deleted": submission_id}


@app.get("/admin/backups")
def admin_backups(x_admin_key: str = Header(default="")):
    _check_admin(x_admin_key)
    entries = []
    for d in sorted(BACKUPS_DIR.glob("*"), reverse=True):
        if not d.is_dir():
            continue
        files = [f.relative_to(d).as_posix() for f in sorted(d.rglob("*")) if f.is_file()]
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        entries.append({"name": d.name, "files": files, "size": size})
    return {"backups": entries}


@app.get("/admin/backups/{name}/download")
def admin_backup_download(name: str, x_admin_key: str = Header(default="")):
    _check_admin(x_admin_key)
    d = BACKUPS_DIR / name
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Backup not found.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(d))
    buf.seek(0)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="backup_{name}.zip"'},
    )


@app.get("/datasets")
def datasets_page():
    return FileResponse(str(FRONTEND_DIR / "datasets.html"))


@app.get("/datasets/{problem}/manifest")
def dataset_manifest(problem: str):
    _check_problem(problem)
    d = PROBLEMS_DIR / problem
    manifest = {}
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.name != "answer.csv":
            import hashlib as _h
            manifest[f.relative_to(d).as_posix()] = {
                "size": f.stat().st_size,
                "sha256": _h.sha256(f.read_bytes()).hexdigest(),
            }
    return {"problem": problem, "files": manifest}


@app.get("/datasets/{problem}/download")
def dataset_download(problem: str):
    """Public download: train + test ONLY. answer.csv is NEVER included."""
    _check_problem(problem)
    d = PROBLEMS_DIR / problem
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Problem not found.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.name != "answer.csv":
                z.write(f, f.relative_to(d))
    buf.seek(0)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="arise_{problem}_dataset.zip"'},
    )


@app.get("/notebooks/{name}")
def notebook_file(name: str):
    """Serve starter notebook files (public)."""
    safe = Path(name).name
    p = FRONTEND_DIR / "notebooks" / safe
    if not p.is_file() or p.suffix != ".ipynb":
        raise HTTPException(status_code=404, detail="Notebook not found.")
    return FileResponse(str(p), media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
