"""
TrafficEye – number plate OCR.

Steps for one vehicle:
  1. Crop the part of the vehicle box where a plate usually is
     (lower/central area, see config.PLATE_SEARCH_REGIONS).
  2. Convert to grayscale, upscale small crops and boost contrast (CLAHE).
  3. Run EasyOCR. The reader is created once at startup in detector.py and
     passed in, so this module never loads a model itself.
  4. Clean the text and fix common OCR mix-ups (O/0, I/1, S/5, B/8, ...)
     depending on whether the plate format expects a letter or a digit
     at that position.
  5. Accept the text only if it matches an Indian registration format,
     e.g. "TN 33 AB 1234" or the Bharat series "22 BH 1234 AA".

Any failure returns "Not detected". OCR can never crash a request.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterator

import cv2
import numpy as np

from backend import config

logger = logging.getLogger("trafficeye.ocr")

NOT_DETECTED = config.PLATE_NOT_DETECTED
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# State / union-territory codes used on Indian registration plates.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "TS", "UA", "UK", "UP", "WB",
}

# Characters OCR commonly confuses. Applied only where the format demands
# a digit (TO_DIGIT) or a letter (TO_LETTER).
TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}

MAX_CORRECTIONS = 3                     # more fixes than this means the text is probably not a plate
SKIPPED_CHAR_PENALTY = 1.2              # ignoring a character costs more than correcting one

# Standard plate: SS DD LL NNNN = state, district (1-2 digits), series (1-3 letters), number.
STANDARD_LAYOUTS = [(district, series) for district in (2, 1) for series in (2, 1, 3)]
# Bharat series plate: YY BH NNNN LL = year, "BH", number, series (1-2 letters).
BH_SERIES_LENGTHS = (2, 1)


# ---------------------------------------------------------------------------
# Text normalisation (pure Python, easy to unit test)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Uppercase and keep only A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _coerce(chars: str, kind: str) -> tuple[str, int] | None:
    """
    Force every character to a letter (kind 'L') or a digit (kind 'D').
    Returns (fixed_text, number_of_corrections), or None if impossible.
    """
    fixed: list[str] = []
    corrections = 0
    for ch in chars:
        if kind == "L":
            if ch.isalpha():
                fixed.append(ch)
                continue
            if ch in TO_LETTER:
                fixed.append(TO_LETTER[ch])
                corrections += 1
                continue
        else:
            if ch.isdigit():
                fixed.append(ch)
                continue
            if ch in TO_DIGIT:
                fixed.append(TO_DIGIT[ch])
                corrections += 1
                continue
        return None
    return "".join(fixed), corrections


def _split(text: str, lengths: tuple[int, ...]) -> list[str]:
    parts, start = [], 0
    for length in lengths:
        parts.append(text[start:start + length])
        start += length
    return parts


def _standard_candidates(text: str) -> Iterator[tuple[str, int, float]]:
    """Yield (formatted_plate, corrections, penalty) for 'TN 33 AB 1234'-style layouts."""
    for district_len, series_len in STANDARD_LAYOUTS:
        if len(text) != 2 + district_len + series_len + 4:
            continue
        state_raw, district_raw, series_raw, number_raw = _split(text, (2, district_len, series_len, 4))
        pieces = [
            _coerce(state_raw, "L"),
            _coerce(district_raw, "D"),
            _coerce(series_raw, "L"),
            _coerce(number_raw, "D"),
        ]
        if any(piece is None for piece in pieces):
            continue
        (state, f1), (district, f2), (series, f3), (number, f4) = pieces
        if state not in STATE_CODES:
            continue
        corrections = f1 + f2 + f3 + f4
        # Two-digit districts and 1-2 letter series are by far the most common,
        # so rarer layouts cost extra and only win when they need fewer fixes.
        penalty = corrections + (0 if district_len == 2 else 1.0) + {2: 0.0, 1: 0.25, 3: 0.5}[series_len]
        yield f"{state} {district} {series} {number}", corrections, penalty


def _bh_candidates(text: str) -> Iterator[tuple[str, int, float]]:
    """Yield (formatted_plate, corrections, penalty) for '22 BH 1234 AA' layouts."""
    for series_len in BH_SERIES_LENGTHS:
        if len(text) != 2 + 2 + 4 + series_len:
            continue
        year_raw, bh_raw, number_raw, series_raw = _split(text, (2, 2, 4, series_len))
        pieces = [
            _coerce(year_raw, "D"),
            _coerce(bh_raw, "L"),
            _coerce(number_raw, "D"),
            _coerce(series_raw, "L"),
        ]
        if any(piece is None for piece in pieces):
            continue
        (year, f1), (bh, f2), (number, f3), (series, f4) = pieces
        if bh != "BH":
            continue
        corrections = f1 + f2 + f3 + f4
        yield f"{year} BH {number} {series}", corrections, corrections + 0.5


def best_plate_match(text: str) -> tuple[str, float] | None:
    """
    Find the most plausible Indian plate inside `text`.
    Returns (formatted_plate, score) where a lower score is better, or None.
    Sliding windows let us skip noise such as the "IND" mark on HSRP plates.
    """
    cleaned = clean_text(text)
    best: tuple[str, float] | None = None
    for length in range(8, 12):                         # shortest to longest valid layout
        for start in range(0, len(cleaned) - length + 1):
            window = cleaned[start:start + length]
            leftover = len(cleaned) - length              # characters we had to ignore
            for plate, corrections, penalty in (*_standard_candidates(window), *_bh_candidates(window)):
                if corrections > MAX_CORRECTIONS:
                    continue
                score = penalty + SKIPPED_CHAR_PENALTY * leftover
                if best is None or score < best[1]:
                    best = (plate, score)
    return best


def normalize_plate_text(text: str) -> str | None:
    """'tn33 a8 1234' -> 'TN 33 AB 1234'; returns None if no valid plate is found."""
    match = best_plate_match(text)
    return match[0] if match else None


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------
def crop_plate_region(image: np.ndarray, bbox: list[int], vehicle_class: str) -> np.ndarray | None:
    """Cut out the lower/central area of the vehicle where the plate usually sits."""
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    if box_h < config.OCR_MIN_VEHICLE_HEIGHT or box_w < 20:
        return None

    region = config.PLATE_SEARCH_REGIONS.get(vehicle_class, config.PLATE_SEARCH_REGIONS["car"])
    left, top, right, bottom = region
    cx1 = max(0, int(x1 + box_w * left))
    cy1 = max(0, int(y1 + box_h * top))
    cx2 = min(img_w, int(x1 + box_w * right))
    cy2 = min(img_h, int(y1 + box_h * bottom))
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return None
    return image[cy1:cy2, cx1:cx2]


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Grayscale -> resize (upscale small crops, shrink huge ones) -> CLAHE contrast boost."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape[:2]
    if width > config.OCR_MAX_CROP_WIDTH:
        factor = config.OCR_MAX_CROP_WIDTH / width
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    elif height < config.OCR_TARGET_CROP_HEIGHT:
        factor = min(
            config.OCR_MAX_UPSCALE,
            config.OCR_TARGET_CROP_HEIGHT / height,
            config.OCR_MAX_CROP_WIDTH / width,
        )
        if factor > 1:
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _order_fragments(results: list[Any]) -> list[dict[str, Any]]:
    """
    Sort EasyOCR fragments top-to-bottom, then left-to-right within a line.
    Indian motorcycle plates are often two lines ("TN 33" over "AB 1234").
    """
    items = []
    for box, text, confidence in results:
        points = np.asarray(box, dtype=float)
        items.append(
            {
                "text": str(text),
                "confidence": float(confidence),
                "x": float(points[:, 0].min()),
                "y_center": float(points[:, 1].mean()),
                "height": float(points[:, 1].max() - points[:, 1].min()),
            }
        )
    items.sort(key=lambda item: item["y_center"])

    lines: list[list[dict[str, Any]]] = []
    for item in items:
        previous = lines[-1][-1] if lines else None
        same_line = previous is not None and abs(item["y_center"] - previous["y_center"]) < 0.5 * max(
            item["height"], previous["height"], 1.0
        )
        if same_line:
            lines[-1].append(item)
        else:
            lines.append([item])
    return [item for line in lines for item in sorted(line, key=lambda i: i["x"])]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def read_plate(reader: Any, image: np.ndarray, bbox: list[int], vehicle_class: str = "motorcycle") -> dict[str, Any]:
    """
    Read the number plate of one vehicle.
    Returns {"plate_number": "TN 33 AB 1234" | "Not detected", "confidence": float, "raw_text": str}.
    """
    result: dict[str, Any] = {"plate_number": NOT_DETECTED, "confidence": 0.0, "raw_text": ""}
    if reader is None or image is None:
        return result

    try:
        crop = crop_plate_region(image, bbox, vehicle_class)
        if crop is None:
            return result

        raw_results = reader.readtext(preprocess_crop(crop), detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
        fragments = [
            f for f in _order_fragments(raw_results)
            if f["confidence"] >= config.OCR_MIN_TEXT_CONFIDENCE and clean_text(f["text"])
        ]
        if not fragments:
            return result
        result["raw_text"] = " ".join(f["text"] for f in fragments)

        # Try the whole text (two-line plates), each fragment alone and each
        # neighbouring pair, then keep the most plausible plate.
        candidates = [(result["raw_text"], float(np.mean([f["confidence"] for f in fragments])))]
        candidates += [(f["text"], f["confidence"]) for f in fragments]
        candidates += [
            (a["text"] + b["text"], (a["confidence"] + b["confidence"]) / 2)
            for a, b in zip(fragments, fragments[1:])
        ]

        best: tuple[str, float, float] | None = None       # (plate, score, ocr_confidence)
        for text, confidence in candidates:
            match = best_plate_match(text)
            if match is None:
                continue
            plate, score = match
            if best is None or score < best[1] or (score == best[1] and confidence > best[2]):
                best = (plate, score, confidence)

        if best is not None:
            result["plate_number"] = best[0]
            result["confidence"] = round(best[2], 3)
    except Exception:
        logger.exception("Plate OCR failed - continuing without a plate number")
    return result
