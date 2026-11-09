"""ARISE 2026 — ORM models."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from database import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True)
    participant_id = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    school = Column(String, nullable=True)
    email = Column(String, nullable=True)
    token = Column(String, nullable=False)
    created_at = Column(DateTime, default=_utcnow)


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    participant_id = Column(String, index=True, nullable=False)
    problem = Column(String, index=True, nullable=False, default="explorer")
    filename = Column(String, nullable=False)
    md5 = Column(String, nullable=True, index=True)
    status = Column(String, nullable=False)  # accepted | rejected
    error = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    rows = Column(Integer, nullable=True)
    submitted_at = Column(DateTime, default=_utcnow, index=True)
