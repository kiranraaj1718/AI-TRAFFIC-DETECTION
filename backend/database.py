"""
TrafficEye – SQLite database (SQLAlchemy 2.x).

One table, `violations`, holds every detected violation:

    id, type, confidence, plate_number, fine,
    evidence_path, report_path, source_file, created_at

evidence_path and report_path are stored as "<folder>/<file name>"
(e.g. "evidence/evidence_20260925_101500_ab12cd34ef.jpg"), never as absolute
paths, so the database keeps working if the project is moved or copied to
another machine.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend import config

logger = logging.getLogger("trafficeye.database")

# check_same_thread=False: FastAPI serves requests from a thread pool, and every
# request opens its own short-lived session, so sharing the engine is safe.
engine = create_engine(
    f"sqlite:///{config.DATABASE_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ViolationRecord(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(32), index=True)                  # NO_HELMET / TRIPLE_RIDING
    confidence: Mapped[float] = mapped_column(Float)                            # 0.0 - 1.0
    plate_number: Mapped[str] = mapped_column(String(32), index=True, default=config.PLATE_NOT_DETECTED)
    fine: Mapped[int] = mapped_column(Integer)                                  # rupees
    evidence_path: Mapped[str] = mapped_column(String(255), index=True)        # "evidence/<file>.jpg"
    report_path: Mapped[str | None] = mapped_column(String(255), nullable=True)  # "reports/challan_<id>.pdf"
    source_file: Mapped[str] = mapped_column(String(255))                      # original upload name
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

    @property
    def challan_id(self) -> str:
        """Human-friendly challan number, e.g. TE-20260925-00042."""
        return f"TE-{self.created_at:%Y%m%d}-{self.id:05d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "challan_id": self.challan_id,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "plate_number": self.plate_number,
            "fine": self.fine,
            "evidence_path": self.evidence_path,
            "evidence_url": f"/{self.evidence_path}",
            "report_path": self.report_path,
            "report_url": f"/api/report/{self.id}",
            "source_file": self.source_file,
            "created_at": self.created_at.isoformat(timespec="seconds"),
        }


# ---------------------------------------------------------------------------
# Setup + sessions
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create trafficeye.db and the violations table if they do not exist."""
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    logger.info("Database ready at %s", config.DATABASE_PATH)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a session, commit on success, roll back on any error, always close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Stored-path helpers
# ---------------------------------------------------------------------------
def stored_path(folder: str, filename: str) -> str:
    """Value saved in the database, e.g. stored_path("evidence", "x.jpg") -> "evidence/x.jpg"."""
    return f"{folder}/{filename}"


def resolve_stored_path(stored: str | None, allowed_dir: Path) -> Path | None:
    """
    "evidence/x.jpg" -> <allowed_dir>/x.jpg.
    Only the final file name is used, so even a tampered database value such as
    "evidence/../../secret.txt" can never point outside `allowed_dir`.
    """
    if not stored:
        return None
    name = PurePosixPath(stored.replace("\\", "/")).name
    if name in ("", ".", ".."):
        return None
    return allowed_dir / name


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def normalize_plate_query(text: str) -> str:
    """'tn 33-ab' -> 'TN33AB' so searches ignore spaces, case and punctuation."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def list_violations(
    session: Session,
    violation_type: str | None = None,
    day: date | None = None,
    search: str | None = None,
    limit: int = config.MAX_HISTORY_RESULTS,
) -> list[ViolationRecord]:
    """Newest first; every filter is optional and they can be combined."""
    stmt = select(ViolationRecord)
    if violation_type:
        stmt = stmt.where(ViolationRecord.type == violation_type)
    if day:
        start = datetime.combine(day, time.min)
        stmt = stmt.where(ViolationRecord.created_at >= start, ViolationRecord.created_at < start + timedelta(days=1))
    if search:
        term = normalize_plate_query(search)
        if term:
            # Compare plates with spaces removed: "TN33" matches "TN 33 AB 1234".
            plate = func.replace(func.upper(ViolationRecord.plate_number), " ", "")
            stmt = stmt.where(plate.contains(term, autoescape=True))
    stmt = stmt.order_by(ViolationRecord.created_at.desc(), ViolationRecord.id.desc()).limit(limit)
    return list(session.scalars(stmt))


def get_stats(session: Session, days: int = 7, recent_limit: int = 5) -> dict[str, Any]:
    """
    Dashboard numbers: totals per type, total fines, a per-day trend for the last
    `days` days (days without violations are filled with 0) and the newest records.
    """
    by_type = {violation_type: 0 for violation_type in config.FINES}
    for violation_type, count in session.execute(
        select(ViolationRecord.type, func.count()).group_by(ViolationRecord.type)
    ):
        by_type[violation_type] = count
    total_fines = session.scalar(select(func.coalesce(func.sum(ViolationRecord.fine), 0)))

    first_day = date.today() - timedelta(days=days - 1)
    day_column = func.date(ViolationRecord.created_at)             # "YYYY-MM-DD" in SQLite
    per_day: dict[str, dict[str, int]] = {}
    for day_text, violation_type, count in session.execute(
        select(day_column, ViolationRecord.type, func.count())
        .where(ViolationRecord.created_at >= datetime.combine(first_day, time.min))
        .group_by(day_column, ViolationRecord.type)
    ):
        per_day.setdefault(day_text, {})[violation_type] = count

    daily_trend = []
    for offset in range(days):
        day_text = (first_day + timedelta(days=offset)).isoformat()
        counts = per_day.get(day_text, {})
        daily_trend.append(
            {
                "date": day_text,
                "count": sum(counts.values()),
                "no_helmet": counts.get("NO_HELMET", 0),
                "triple_riding": counts.get("TRIPLE_RIDING", 0),
            }
        )

    recent = session.scalars(
        select(ViolationRecord).order_by(ViolationRecord.created_at.desc(), ViolationRecord.id.desc()).limit(recent_limit)
    )
    return {
        "total": sum(by_type.values()),
        "no_helmet": by_type.get("NO_HELMET", 0),
        "triple_riding": by_type.get("TRIPLE_RIDING", 0),
        "total_fines": int(total_fines or 0),
        "today": daily_trend[-1]["count"],
        "by_type": by_type,
        "daily_trend": daily_trend,
        "recent": [record.to_dict() for record in recent],
    }


def count_evidence_references(session: Session, evidence_path: str) -> int:
    """How many violations point at an evidence image (one image can hold several violations)."""
    stmt = select(func.count()).select_from(ViolationRecord).where(ViolationRecord.evidence_path == evidence_path)
    return int(session.scalar(stmt) or 0)
