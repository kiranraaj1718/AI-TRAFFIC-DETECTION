"""
TrafficEye – demo data seeder.

    python -m backend.seed          # (re)create ~20 demo violations
    python -m backend.seed --clear  # remove the demo violations only

Inserts 20 realistic violation records (13 no-helmet, 7 triple riding) spread
over the last 7 days, with Tamil Nadu / Indian plate numbers. Each incident gets
a synthetic evidence image drawn with OpenCV (clearly marked as a demo image)
and every record gets a PDF e-challan, so the dashboard and history pages look
populated during a demo.

Three incidents have BOTH violations on the same bike, so two records share one
evidence image – handy for showing that deleting one keeps the shared image.

Re-running the script first removes the previous demo records (their
source_file starts with "demo_seed_"). Real detections are never touched.
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from datetime import datetime, time, timedelta

import cv2
import numpy as np
from sqlalchemy import select

from backend import config, database, report
from backend.database import ViolationRecord, session_scope
from backend.detector import add_evidence_footer, draw_box, save_evidence

logger = logging.getLogger("trafficeye.seed")

RNG_SEED = 2026
NOT_DETECTED = config.PLATE_NOT_DETECTED

# Realistic plates: mostly Tamil Nadu RTOs plus a few neighbouring states.
PLATES = [
    "TN 01 AB 4521", "TN 09 BK 7812", "TN 22 CX 3390", "TN 33 AB 1234", "TN 37 DM 5521",
    "TN 38 BZ 9014", "TN 45 AR 6630", "TN 58 AJ 2187", "TN 59 BC 4410", "TN 66 H 2231",
    "TN 72 CE 8015", "TN 07 CQ 1108", "KA 01 MN 1234", "KL 07 CD 5678", "PY 01 AK 3345",
    NOT_DETECTED, NOT_DETECTED,
]

# 17 incidents -> 20 records: 3 with both violations, 4 triple riding only, 10 no helmet only.
INCIDENTS = (
    [("TRIPLE_RIDING", "NO_HELMET")] * 3
    + [("TRIPLE_RIDING",)] * 4
    + [("NO_HELMET",)] * 10
)

RED, GREEN = config.COLOR_VIOLATION, config.COLOR_NORMAL


# ---------------------------------------------------------------------------
# Synthetic evidence image
# ---------------------------------------------------------------------------
def _draw_background(img: np.ndarray, rng: random.Random) -> int:
    """Dusk sky, city silhouette, asphalt road with lane markings. Returns horizon y."""
    h, w = img.shape[:2]
    horizon = int(h * 0.38)
    for y in range(horizon):
        t = y / horizon
        img[y, :] = (int(95 - 35 * t), int(55 - 10 * t), int(35 + 25 * t))       # BGR gradient
    for x in range(0, w, 60):                                                   # buildings
        top = horizon - rng.randint(30, 120)
        cv2.rectangle(img, (x, top), (x + rng.randint(40, 58), horizon), (38, 32, 30), -1)
        for wy in range(top + 10, horizon - 8, 18):                             # lit windows
            if rng.random() < 0.35:
                cv2.rectangle(img, (x + 10, wy), (x + 18, wy + 8), (90, 190, 235), -1)
    img[horizon:, :] = (62, 60, 58)
    vanishing = (w // 2, horizon)
    cv2.line(img, (int(w * 0.02), h), vanishing, (200, 200, 200), 3, cv2.LINE_AA)
    cv2.line(img, (int(w * 0.98), h), vanishing, (200, 200, 200), 3, cv2.LINE_AA)
    for k in range(7):                                                          # dashed centre line
        t0, t1 = (k / 7) ** 1.7, ((k + 0.45) / 7) ** 1.7
        y0, y1 = int(horizon + (h - horizon) * t0), int(horizon + (h - horizon) * t1)
        cv2.line(img, (w // 2, y0), (w // 2, y1), (40, 200, 230), max(1, int(2 + 6 * t1)), cv2.LINE_AA)
    return horizon


def _draw_motorcycle(img: np.ndarray, cx: int, base_y: int, color: tuple[int, int, int]) -> None:
    for wheel_x in (cx - 95, cx + 95):
        cv2.circle(img, (wheel_x, base_y - 38), 38, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(img, (wheel_x, base_y - 38), 38, (150, 150, 150), 5, cv2.LINE_AA)
        cv2.circle(img, (wheel_x, base_y - 38), 8, (180, 180, 180), -1, cv2.LINE_AA)
    body = np.array(
        [(cx - 100, base_y - 72), (cx - 25, base_y - 112), (cx + 60, base_y - 108),
         (cx + 105, base_y - 74), (cx + 40, base_y - 58), (cx - 60, base_y - 58)],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [body], color, cv2.LINE_AA)
    cv2.line(img, (cx + 75, base_y - 110), (cx + 100, base_y - 145), (60, 60, 60), 6, cv2.LINE_AA)   # handlebar
    cv2.rectangle(img, (cx - 70, base_y - 122), (cx + 40, base_y - 108), (30, 30, 30), -1)          # seat


def _draw_plate(img: np.ndarray, cx: int, base_y: int, plate: str) -> None:
    text = plate if plate != NOT_DETECTED else "TN ?? ?? ????"
    cv2.rectangle(img, (cx - 150, base_y - 98), (cx - 62, base_y - 74), (245, 245, 245), -1)
    cv2.rectangle(img, (cx - 150, base_y - 98), (cx - 62, base_y - 74), (20, 20, 20), 1)
    cv2.putText(img, text, (cx - 147, base_y - 81), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_rider(img: np.ndarray, x: int, seat_y: int, shirt: tuple[int, int, int], helmet: bool) -> None:
    """Rider sitting at seat_y (the bottom of the torso)."""
    cv2.line(img, (x - 8, seat_y), (x - 25, seat_y + 50), (60, 45, 40), 9, cv2.LINE_AA)            # leg
    cv2.rectangle(img, (x - 20, seat_y - 95), (x + 20, seat_y), shirt, -1)                          # torso
    head = (x, seat_y - 118)
    if helmet:
        cv2.circle(img, head, 21, (35, 35, 170), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x + 6, seat_y - 116), (12, 8), 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)  # visor
    else:
        cv2.circle(img, head, 18, (120, 160, 205), -1, cv2.LINE_AA)                                 # face
        cv2.ellipse(img, (x, seat_y - 126), (18, 10), 0, 180, 360, (25, 25, 30), -1, cv2.LINE_AA)   # hair


def draw_demo_evidence(
    rng: random.Random, types: tuple[str, ...], confidences: dict[str, float], plate: str, taken_at: datetime,
) -> np.ndarray:
    """Draw a cartoon traffic scene annotated the same way real evidence is."""
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    _draw_background(img, rng)

    cx, base_y = 480 + rng.randint(-160, 160), 520
    _draw_motorcycle(img, cx, base_y, rng.choice([(40, 40, 180), (160, 90, 30), (30, 30, 30), (20, 120, 20)]))
    _draw_plate(img, cx, base_y, plate)

    triple = "TRIPLE_RIDING" in types
    no_helmet = "NO_HELMET" in types
    rider_count = 3 if triple else rng.choice([1, 2])
    # The bike faces right: the driver is rightmost, pillions sit behind and a
    # little higher, which also keeps each rider's label tag at its own height.
    xs = [cx + 45 - (rider_count - 1 - i) * 58 for i in range(rider_count)]
    seats = [base_y - 112 - (rider_count - 1 - i) * 26 for i in range(rider_count)]
    # At least one rider is bare-headed for no-helmet incidents; otherwise everyone wears one.
    bare = set(rng.sample(range(rider_count), k=rng.randint(1, rider_count))) if no_helmet else set()
    shirts = [(180, 120, 40), (60, 140, 220), (90, 90, 90), (150, 60, 140)]
    for i, (x, seat_y) in enumerate(zip(xs, seats)):
        _draw_rider(img, x, seat_y, rng.choice(shirts), helmet=i not in bare)

    # Annotate exactly like the live pipeline does.
    font_scale, thickness = 0.55, 2
    labels = [f"{t.replace('_', ' ')} {confidences[t]:.0%}" for t in types]
    for i, (x, seat_y) in enumerate(zip(xs, seats)):
        violating = triple or i in bare
        label = f"NO HELMET {confidences['NO_HELMET']:.0%}" if i in bare else f"helmet {rng.uniform(0.8, 0.95):.0%}"
        box = [x - 28, seat_y - 144, x + 28, seat_y + 52]
        draw_box(img, box, RED if violating else GREEN, [label], font_scale, thickness)
    draw_box(
        img, [cx - 155, base_y - 150, cx + 150, base_y - 2], RED, labels, font_scale, thickness,
        bottom_label=None if plate == NOT_DETECTED else plate,
    )

    cv2.rectangle(img, (10, 10), (330, 38), (20, 20, 20), -1)
    cv2.putText(img, "SYNTHETIC DEMO IMAGE - TrafficEye seed", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return add_evidence_footer(img, f"TrafficEye evidence | {taken_at:%Y-%m-%d %H:%M:%S} | demo data (synthetic)")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def clear_demo_data() -> int:
    """Delete previous demo records and their challans / evidence images."""
    with session_scope() as session:
        demo_records = list(
            session.scalars(select(ViolationRecord).where(ViolationRecord.source_file.startswith(config.DEMO_SOURCE_PREFIX)))
        )
        files = [(r.id, r.evidence_path, r.report_path) for r in demo_records]
        for record in demo_records:
            session.delete(record)
        session.flush()
        still_used = {path for _, path, _ in files if database.count_evidence_references(session, path) > 0}

    for violation_id, evidence_path, report_path in files:
        for pdf in {database.resolve_stored_path(report_path, config.REPORT_DIR), report.challan_path(violation_id)}:
            if pdf is not None and pdf.is_file():
                pdf.unlink()
        evidence = database.resolve_stored_path(evidence_path, config.EVIDENCE_DIR)
        if evidence_path not in still_used and evidence is not None and evidence.is_file():
            evidence.unlink()
    return len(files)


def _incident_time(rng: random.Random, days_ago: int, now: datetime) -> datetime:
    """A plausible daytime timestamp `days_ago` days back (never in the future)."""
    day = (now - timedelta(days=days_ago)).date()
    moment = datetime.combine(day, time(7, 30)) + timedelta(minutes=rng.randint(0, 14 * 60))
    if moment > now:
        minutes_today = now.hour * 60 + now.minute
        moment = now - timedelta(minutes=rng.randint(0, max(0, min(minutes_today - 1, 180))))
    return moment.replace(microsecond=0)


def seed() -> list[ViolationRecord]:
    rng = random.Random(RNG_SEED)
    now = datetime.now()

    incidents = INCIDENTS[:]
    rng.shuffle(incidents)
    plates = PLATES[:]
    rng.shuffle(plates)
    days = [i % 7 for i in range(len(incidents))]      # every one of the last 7 days gets data
    rng.shuffle(days)

    records: list[ViolationRecord] = []
    for number, (types, plate, days_ago) in enumerate(zip(incidents, plates, days), start=1):
        taken_at = _incident_time(rng, days_ago, now)
        confidences = {vtype: round(rng.uniform(0.74, 0.97), 3) for vtype in types}
        evidence_name = save_evidence(draw_demo_evidence(rng, types, confidences, plate, taken_at), prefix="demo")
        for vtype in types:
            records.append(
                ViolationRecord(
                    type=vtype,
                    confidence=confidences[vtype],
                    plate_number=plate,
                    fine=config.FINES[vtype],
                    evidence_path=database.stored_path("evidence", evidence_name),
                    source_file=f"{config.DEMO_SOURCE_PREFIX}{number:02d}.jpg",
                    created_at=taken_at,
                )
            )

    with session_scope() as session:
        session.add_all(records)
        session.flush()                                  # assigns ids needed for challan names
        for record in records:
            try:
                record.report_path = report.generate_challan(record)
            except report.ReportError:
                logger.warning("Challan for demo violation %s will be generated on first download", record.id)
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Populate TrafficEye with demo violations.")
    parser.add_argument("--clear", action="store_true", help="only remove previously seeded demo data")
    args = parser.parse_args()

    config.ensure_directories()
    database.init_db()

    removed = clear_demo_data()
    if removed:
        logger.info("Removed %d previous demo records", removed)
    if args.clear:
        print(f"Removed {removed} demo violations. Real detections were not touched.")
        return

    records = seed()
    by_type = Counter(r.type for r in records)
    evidence_images = len({r.evidence_path for r in records})
    challans = sum(1 for r in records if r.report_path)
    first, last = min(r.created_at for r in records), max(r.created_at for r in records)
    print(
        f"Inserted {len(records)} demo violations "
        f"({by_type['NO_HELMET']} NO_HELMET, {by_type['TRIPLE_RIDING']} TRIPLE_RIDING) "
        f"from {first:%d %b} to {last:%d %b %Y}.\n"
        f"Evidence images: {evidence_images} (3 shared by two violations). PDF challans: {challans}.\n"
        f"Total fines: Rs. {sum(r.fine for r in records):,}. Start the server and open http://localhost:8000"
    )


if __name__ == "__main__":
    main()
