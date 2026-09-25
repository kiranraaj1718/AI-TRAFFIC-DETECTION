"""
TrafficEye – model loading, object detection and the violation engine.

Pipeline for one image / video frame (analyze_frame):
  1. YOLOv8 finds people, motorcycles, cars, buses and trucks.
  2. Each person is linked to at most one motorcycle (rider association).
  3. If helmet.pt is loaded, helmet / no-helmet boxes are matched to each
     rider's head region (top 30% of the person box), giving every rider
     the status HELMET, NO_HELMET or UNKNOWN.
  4. Rules turn that into violations: TRIPLE_RIDING and NO_HELMET.
  5. EasyOCR reads the number plate of every violating motorcycle.
process_image() then draws the evidence image (red = violation, green = normal).

Anything that can fail (model download, corrupt weights, inference) is wrapped
in try/except so the API can report a useful error instead of crashing.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any

import cv2
import numpy as np

from backend import config, ocr

logger = logging.getLogger("trafficeye.detector")

FONT = cv2.FONT_HERSHEY_SIMPLEX
TAG_PADDING = 4

# Rider helmet status
HELMET = "HELMET"
NO_HELMET = "NO_HELMET"
UNKNOWN = "UNKNOWN"

# Violation types
VIOLATION_NO_HELMET = "NO_HELMET"
VIOLATION_TRIPLE_RIDING = "TRIPLE_RIDING"

# Helmet check outcome reported with every analysis
HELMET_CHECK_ENABLED = "enabled"      # helmet model ran on this image
HELMET_CHECK_SKIPPED = "skipped"      # helmet.pt not loaded
HELMET_CHECK_FAILED = "failed"        # helmet model crashed on this image


# ---------------------------------------------------------------------------
# Errors raised to the API layer
# ---------------------------------------------------------------------------
class ModelNotLoadedError(RuntimeError):
    """A request needs a model that failed to load at startup."""


class DetectionError(RuntimeError):
    """YOLO inference or evidence generation failed."""


# ---------------------------------------------------------------------------
# Model registry – one shared instance holds every loaded model
# ---------------------------------------------------------------------------
@dataclass
class ModelRegistry:
    yolo: Any = None                    # ultralytics.YOLO trained on COCO (yolov8n.pt)
    helmet: Any = None                  # ultralytics.YOLO helmet model (optional)
    ocr_reader: Any = None              # easyocr.Reader
    device: str = "cpu"                 # "cpu" or "cuda:0"
    yolo_class_ids: list[int] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    # Ultralytics and EasyOCR objects are not guaranteed to be thread-safe.
    # FastAPI runs our blocking inference in a thread pool, so a lock makes
    # sure only one request uses a model at a time.
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def yolo_loaded(self) -> bool:
        return self.yolo is not None

    @property
    def helmet_loaded(self) -> bool:
        return self.helmet is not None

    @property
    def ocr_loaded(self) -> bool:
        return self.ocr_reader is not None

    def helmet_class_mapping(self) -> dict[str, str]:
        """How each helmet-model class is interpreted: HELMET, NO_HELMET or IGNORED."""
        if self.helmet is None:
            return {}
        return {name: normalize_helmet_class(name) or "IGNORED" for name in self.helmet.names.values()}

    def status(self) -> dict[str, Any]:
        """Summary used by GET /api/health and the navbar status dots."""
        helmet_classes = list(self.helmet.names.values()) if self.helmet is not None else []
        return {
            "yolo_loaded": self.yolo_loaded,
            "helmet_model_loaded": self.helmet_loaded,
            "ocr_loaded": self.ocr_loaded,
            "device": self.device,
            "helmet_classes": helmet_classes,
            "helmet_class_mapping": self.helmet_class_mapping(),
            "errors": dict(self.errors),
        }


models = ModelRegistry()


# ---------------------------------------------------------------------------
# Startup loading
# ---------------------------------------------------------------------------
def select_device() -> str:
    """Use CUDA only when PyTorch can actually see a GPU; CPU otherwise."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception as exc:  # torch missing or broken -> CPU is the safe default
        logger.warning("Could not query CUDA availability (%s); using CPU", exc)
    return "cpu"


def load_models() -> ModelRegistry:
    """Load every model once. Each loader records its own success or failure."""
    models.errors.clear()
    models.device = select_device()
    logger.info("Inference device: %s", models.device)

    _load_yolo()
    _load_helmet_model()
    _load_ocr_reader()

    logger.info(
        "Model status -> YOLO: %s | Helmet: %s | OCR: %s",
        "loaded" if models.yolo_loaded else "FAILED",
        "loaded" if models.helmet_loaded else "not loaded",
        "loaded" if models.ocr_loaded else "FAILED",
    )
    return models


def unload_models() -> None:
    """Release model references on shutdown."""
    models.yolo = None
    models.helmet = None
    models.ocr_reader = None
    models.yolo_class_ids = []


def _warm_up(model: Any) -> None:
    """Run one dummy prediction so the first real request is not slow."""
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, device=models.device, verbose=False)


def _load_yolo() -> None:
    try:
        from ultralytics import YOLO

        started = time.perf_counter()
        # Giving a path inside backend/models makes Ultralytics download
        # yolov8n.pt there on the first run instead of the working directory.
        model = YOLO(str(config.YOLO_MODEL_PATH))
        class_ids = [cid for cid, name in model.names.items() if name in config.TARGET_CLASSES]
        if not class_ids:
            raise RuntimeError(f"model has none of the target classes {config.TARGET_CLASSES}")
        _warm_up(model)

        models.yolo = model
        models.yolo_class_ids = class_ids
        logger.info("YOLOv8 loaded in %.1fs (tracking class ids %s)", time.perf_counter() - started, class_ids)
    except Exception as exc:
        models.yolo = None
        models.errors["yolo"] = f"Failed to load {config.YOLO_MODEL_PATH.name}: {exc}"
        logger.exception("YOLOv8 failed to load")


def _load_helmet_model() -> None:
    path = config.HELMET_MODEL_PATH
    if not path.is_file():
        models.errors["helmet"] = "backend/models/helmet.pt not found - helmet check skipped"
        logger.warning("Helmet model not found at %s - helmet checks are disabled", path)
        return
    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        if model.task != "detect":
            raise RuntimeError(f"helmet.pt must be a YOLO detection model (task 'detect'), got '{model.task}'")
        _warm_up(model)

        models.helmet = model
        mapping = models.helmet_class_mapping()
        logger.info("Helmet model loaded. Class mapping: %s", mapping)
        if NO_HELMET not in mapping.values():
            logger.warning(
                "No helmet.pt class maps to NO_HELMET, so no-helmet violations cannot be raised. "
                "Add the class name to NO_HELMET_CLASS_KEYS in config.py."
            )
    except Exception as exc:
        models.helmet = None
        models.errors["helmet"] = f"helmet.pt could not be loaded: {exc}"
        logger.exception("Helmet model failed to load")


OCR_LOAD_ATTEMPTS = 3


def _load_ocr_reader() -> None:
    try:
        import easyocr

        started = time.perf_counter()
        config.OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Loading EasyOCR (the first run downloads about 100 MB of weights)...")
        # On Windows, antivirus scanning of the freshly downloaded weights zip can
        # make EasyOCR's clean-up fail ("file is being used by another process")
        # even though the weights were extracted. A retry then loads from disk.
        for attempt in range(1, OCR_LOAD_ATTEMPTS + 1):
            try:
                with warnings.catch_warnings():
                    # EasyOCR's CPU speed-up uses a PyTorch quantisation API that
                    # now prints a deprecation warning; it is harmless noise.
                    warnings.filterwarnings("ignore", message=".*quantize.*", category=UserWarning)
                    reader = easyocr.Reader(
                        config.OCR_LANGUAGES,
                        gpu=models.device.startswith("cuda"),
                        model_storage_directory=str(config.OCR_MODEL_DIR),
                        user_network_directory=str(config.OCR_MODEL_DIR / "user_network"),
                        verbose=False,
                    )
                break
            except Exception as exc:
                if attempt == OCR_LOAD_ATTEMPTS:
                    raise
                logger.warning("EasyOCR load attempt %d failed (%s) - retrying in 3s", attempt, exc)
                time.sleep(3)
        models.ocr_reader = reader
        logger.info("EasyOCR loaded in %.1fs", time.perf_counter() - started)
    except Exception as exc:
        models.ocr_reader = None
        models.errors["ocr"] = f"EasyOCR failed to load: {exc}"
        logger.exception("EasyOCR failed to load")


# ---------------------------------------------------------------------------
# Box geometry helpers (boxes are [x1, y1, x2, y2])
# ---------------------------------------------------------------------------
def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: list[float], b: list[float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over Union: 0 = no overlap, 1 = identical boxes."""
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box: list[float], x_ratio: float, y_ratio: float) -> list[float]:
    """Grow a box by a fraction of its width/height on every side."""
    width, height = box[2] - box[0], box[3] - box[1]
    return [box[0] - width * x_ratio, box[1] - height * y_ratio, box[2] + width * x_ratio, box[3] + height * y_ratio]


def box_center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def point_in_box(x: float, y: float, box: list[float]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def head_region(person_box: list[int]) -> list[int]:
    """The top HEAD_REGION_RATIO (30%) of a person box is where the head/helmet is."""
    x1, y1, x2, y2 = person_box
    head_height = max(1, int(round((y2 - y1) * config.HEAD_REGION_RATIO)))
    return [x1, y1, x2, y1 + head_height]


def clip_bbox(bbox: Any, width: int, height: int) -> list[int]:
    """Round a box to integer pixels and keep it inside the image."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Data structures produced by the violation engine
# ---------------------------------------------------------------------------
@dataclass
class Rider:
    person_index: int                   # index into the detections list
    bbox: list[int]
    confidence: float
    association_score: float
    helmet_status: str = UNKNOWN
    helmet_confidence: float | None = None
    helmet_bbox: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "helmet_status": self.helmet_status,
            "helmet_confidence": self.helmet_confidence,
        }


@dataclass
class MotorcycleGroup:
    index: int                          # index of the motorcycle in the detections list
    bbox: list[int]
    confidence: float
    riders: list[Rider] = field(default_factory=list)
    plate: dict[str, Any] | None = None     # OCR result (only read for violating motorcycles)
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "rider_count": len(self.riders),
            "riders": [rider.to_dict() for rider in self.riders],
            "plate_number": self.plate["plate_number"] if self.plate else None,
            "violations": [v["type"] for v in self.violations],
        }


@dataclass
class FrameAnalysis:
    detections: list[dict[str, Any]]
    helmet_detections: list[dict[str, Any]]
    motorcycles: list[MotorcycleGroup]
    violations: list[dict[str, Any]]
    helmet_check: str

    def to_response(self) -> dict[str, Any]:
        return {
            "detections": self.detections,
            "object_counts": count_objects(self.detections),
            "motorcycles": [group.to_dict() for group in self.motorcycles],
            "violations": self.violations,
            "helmet_model_loaded": models.helmet_loaded,
            "helmet_check": self.helmet_check,
            "ocr_loaded": models.ocr_loaded,
        }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def resize_for_inference(image: np.ndarray, max_width: int = config.MAX_FRAME_WIDTH) -> np.ndarray:
    """Downscale wide images so inference, drawing and storage stay fast."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def detect_objects(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Run YOLOv8 and return detections of the target classes, e.g.
        {"class": "motorcycle", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
    sorted by confidence (highest first).
    """
    if models.yolo is None:
        raise ModelNotLoadedError(models.errors.get("yolo", "YOLOv8 model is not loaded"))

    try:
        with models.lock:
            results = models.yolo.predict(
                source=image,
                conf=config.YOLO_CONFIDENCE,
                iou=config.IOU_THRESHOLD,
                imgsz=config.YOLO_IMAGE_SIZE,
                classes=models.yolo_class_ids,
                device=models.device,
                verbose=False,
            )
    except Exception as exc:
        logger.exception("YOLO inference failed")
        raise DetectionError(f"YOLO inference failed: {exc}") from exc

    if not results:
        return []
    return _parse_boxes(results[0], models.yolo.names)


def _parse_boxes(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    """Convert an Ultralytics result into plain Python dicts."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = result.orig_shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    detections = []
    for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
        detections.append(
            {
                "class": names.get(int(class_id), str(class_id)),
                "confidence": round(float(confidence), 3),
                "bbox": clip_bbox(bbox, width, height),
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def count_objects(detections: list[dict[str, Any]]) -> dict[str, int]:
    """Count detections per target class (classes with zero hits are included)."""
    counts = {name: 0 for name in config.TARGET_CLASSES}
    for det in detections:
        counts[det["class"]] = counts.get(det["class"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Helmet detection
# ---------------------------------------------------------------------------
def normalize_helmet_class(name: str) -> str | None:
    """
    Map a helmet-model class name to HELMET / NO_HELMET, or None to ignore it.
    "With Helmet", "with_helmet", "helmet"          -> HELMET
    "no-helmet", "Without Helmet", "head"           -> NO_HELMET
    """
    key = re.sub(r"[^a-z]", "", str(name).lower())
    if key in config.NO_HELMET_CLASS_KEYS:
        return NO_HELMET
    if key in config.HELMET_CLASS_KEYS:
        return HELMET
    return None


def detect_helmets(image: np.ndarray) -> tuple[list[dict[str, Any]], str]:
    """
    Run the optional helmet model on the whole image.
    Returns (helmet_detections, helmet_check) where each detection also carries
    "status": HELMET / NO_HELMET. A helmet-model failure never fails the request;
    the check is just reported as "failed" and every rider stays UNKNOWN.
    """
    if models.helmet is None:
        return [], HELMET_CHECK_SKIPPED

    try:
        with models.lock:
            results = models.helmet.predict(
                source=image,
                conf=config.HELMET_CONFIDENCE,
                iou=config.IOU_THRESHOLD,
                imgsz=config.YOLO_IMAGE_SIZE,
                device=models.device,
                verbose=False,
            )
    except Exception:
        logger.exception("Helmet inference failed - helmet check skipped for this image")
        return [], HELMET_CHECK_FAILED

    detections = []
    for det in _parse_boxes(results[0], models.helmet.names) if results else []:
        status = normalize_helmet_class(det["class"])
        if status is not None:
            detections.append({**det, "status": status})
    return detections, HELMET_CHECK_ENABLED


# ---------------------------------------------------------------------------
# Violation engine
# ---------------------------------------------------------------------------
def rider_match_score(person: list[int], motorcycle: list[int]) -> float | None:
    """
    How well a person box fits as a rider of a motorcycle box.
    Returns None when the person cannot be a rider, otherwise a score where
    higher = better (more overlap, better centred).
    """
    px1, py1, px2, py2 = person
    mx1, my1, mx2, my2 = motorcycle
    person_h = py2 - py1
    moto_w, moto_h = mx2 - mx1, my2 - my1
    if person_h <= 0 or moto_w <= 0 or moto_h <= 0:
        return None

    # Size sanity: a rider is roughly 0.5x - 3.5x as tall as the motorcycle box.
    height_ratio = person_h / moto_h
    if not config.MIN_RIDER_HEIGHT_RATIO <= height_ratio <= config.MAX_RIDER_HEIGHT_RATIO:
        return None

    # A rider's head is above the middle of the motorcycle.
    if py1 > my1 + 0.5 * moto_h:
        return None

    expanded = expand_box(motorcycle, config.RIDER_BOX_EXPAND_X, config.RIDER_BOX_EXPAND_Y)
    person_cx = (px1 + px2) / 2
    # The rider must be horizontally over the (slightly widened) motorcycle.
    if not expanded[0] <= person_cx <= expanded[2]:
        return None

    overlap = intersection_area(person, motorcycle) / box_area(person)   # share of the person on the bike
    bottom_center_inside = point_in_box(person_cx, py2, expanded)
    if overlap < config.MIN_RIDER_OVERLAP and not bottom_center_inside:
        return None

    centre_offset = min(1.0, abs(person_cx - (mx1 + mx2) / 2) / moto_w)
    return overlap + iou(person, motorcycle) + (0.3 if bottom_center_inside else 0.0) + 0.3 * (1.0 - centre_offset)


def associate_riders(detections: list[dict[str, Any]]) -> list[MotorcycleGroup]:
    """
    Link every person to the single best-matching motorcycle (or to none).
    Because each person picks only their best match, nobody is counted on two bikes.
    """
    groups = [
        MotorcycleGroup(index=i, bbox=det["bbox"], confidence=det["confidence"])
        for i, det in enumerate(detections)
        if det["class"] == "motorcycle"
    ]
    if not groups:
        return []

    for person_index, det in enumerate(detections):
        if det["class"] != "person":
            continue
        best_group, best_score = None, 0.0
        for group in groups:
            score = rider_match_score(det["bbox"], group.bbox)
            if score is not None and (best_group is None or score > best_score):
                best_group, best_score = group, score
        if best_group is not None:
            best_group.riders.append(
                Rider(
                    person_index=person_index,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    association_score=round(best_score, 3),
                )
            )

    for group in groups:
        group.riders.sort(key=lambda rider: rider.bbox[0])       # left to right
    return groups


def helmet_match_score(helmet_box: list[int], head_box: list[int]) -> float | None:
    """How well a helmet/no-helmet box fits a rider's head region (None = no match)."""
    helmet_area = box_area(helmet_box)
    if helmet_area <= 0 or helmet_area > 2.5 * max(1.0, box_area(head_box)):
        return None                                   # far bigger than a head: not this rider's
    containment = intersection_area(helmet_box, head_box) / helmet_area
    cx, cy = box_center(helmet_box)
    center_inside = point_in_box(cx, cy, expand_box(head_box, 0.10, 0.15))
    if containment < config.HEAD_MATCH_MIN_OVERLAP and not center_inside:
        return None
    # Tie-breaker for overlapping riders: prefer the head whose centre is closest.
    hx, hy = box_center(head_box)
    head_size = max(1.0, head_box[2] - head_box[0], head_box[3] - head_box[1])
    closeness = 1.0 - min(1.0, ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 / head_size)
    return containment + (0.5 if center_inside else 0.0) + 0.2 * closeness


def assign_helmet_status(groups: list[MotorcycleGroup], helmet_detections: list[dict[str, Any]]) -> None:
    """
    Give each rider the status of the best-matching helmet/no-helmet box.
    Greedy one-to-one matching: each box is used for at most one rider, so two
    overlapping riders cannot share the same head. Unmatched riders stay UNKNOWN.
    """
    riders = [rider for group in groups for rider in group.riders]
    pairs = []
    for rider_idx, rider in enumerate(riders):
        head = head_region(rider.bbox)
        for det_idx, det in enumerate(helmet_detections):
            score = helmet_match_score(det["bbox"], head)
            if score is not None:
                pairs.append((score, det["confidence"], rider_idx, det_idx))

    pairs.sort(reverse=True)
    used_riders: set[int] = set()
    used_detections: set[int] = set()
    for _, _, rider_idx, det_idx in pairs:
        if rider_idx in used_riders or det_idx in used_detections:
            continue
        rider, det = riders[rider_idx], helmet_detections[det_idx]
        rider.helmet_status = det["status"]
        rider.helmet_confidence = det["confidence"]
        rider.helmet_bbox = det["bbox"]
        used_riders.add(rider_idx)
        used_detections.add(det_idx)


def _make_violation(vtype: str, confidence: float, group: MotorcycleGroup, description: str) -> dict[str, Any]:
    return {
        "type": vtype,
        "confidence": round(float(confidence), 3),
        "fine": config.FINES[vtype],
        "plate_number": ocr.NOT_DETECTED,
        "plate_confidence": 0.0,
        "vehicle_class": "motorcycle",
        "vehicle_bbox": group.bbox,
        "rider_count": len(group.riders),
        "riders_without_helmet": sum(1 for r in group.riders if r.helmet_status == NO_HELMET),
        "rider_bboxes": [r.bbox for r in group.riders],
        "description": description,
    }


def evaluate_violations(group: MotorcycleGroup, helmet_check_ran: bool) -> list[dict[str, Any]]:
    """Apply the traffic rules to one motorcycle and its riders."""
    violations = []
    rider_count = len(group.riders)

    # Rule 1 – triple riding: more than 2 people on one motorcycle.
    if rider_count > config.MAX_RIDERS_PER_MOTORCYCLE:
        confidence = mean([group.confidence] + [r.confidence for r in group.riders])
        violations.append(
            _make_violation(
                VIOLATION_TRIPLE_RIDING,
                confidence,
                group,
                f"{rider_count} riders on one motorcycle (limit {config.MAX_RIDERS_PER_MOTORCYCLE})",
            )
        )

    # Rule 2 – no helmet. Only riders positively detected as NO_HELMET count;
    # UNKNOWN riders are never reported, and nothing is reported without the model.
    if helmet_check_ran:
        without_helmet = [r for r in group.riders if r.helmet_status == NO_HELMET]
        if without_helmet:
            confidence = max(r.helmet_confidence or 0.0 for r in without_helmet)
            violations.append(
                _make_violation(
                    VIOLATION_NO_HELMET,
                    confidence,
                    group,
                    f"{len(without_helmet)} of {rider_count} rider(s) without a helmet",
                )
            )
    return violations


def read_violator_plates(groups: list[MotorcycleGroup], original: np.ndarray, scale_to_original: float) -> None:
    """Run OCR on the plate of every violating motorcycle (capped per image)."""
    if models.ocr_reader is None:
        return
    height, width = original.shape[:2]
    budget = config.OCR_MAX_VEHICLES_PER_IMAGE
    for group in groups:
        if not group.violations or budget <= 0:
            continue
        budget -= 1
        # Boxes are in resized-frame pixels; crop from the full-resolution original.
        bbox = clip_bbox([v * scale_to_original for v in group.bbox], width, height)
        with models.lock:
            group.plate = ocr.read_plate(models.ocr_reader, original, bbox, "motorcycle")
        for violation in group.violations:
            violation["plate_number"] = group.plate["plate_number"]
            violation["plate_confidence"] = group.plate["confidence"]


def analyze_frame(
    frame: np.ndarray,
    original: np.ndarray | None = None,
    scale_to_original: float = 1.0,
    read_plates: bool = True,
) -> FrameAnalysis:
    """
    Full violation analysis of one (already resized) frame.
    `original` is the full-resolution image used for plate OCR.
    """
    detections = detect_objects(frame)
    helmet_detections, helmet_check = detect_helmets(frame)

    motorcycles = associate_riders(detections)
    helmet_check_ran = helmet_check == HELMET_CHECK_ENABLED
    if helmet_check_ran:
        assign_helmet_status(motorcycles, helmet_detections)

    for group in motorcycles:
        group.violations = evaluate_violations(group, helmet_check_ran)

    if read_plates:
        if original is None:
            original, scale_to_original = frame, 1.0
        read_violator_plates(motorcycles, original, scale_to_original)

    violations = [v for group in motorcycles for v in group.violations]
    return FrameAnalysis(detections, helmet_detections, motorcycles, violations, helmet_check)


# ---------------------------------------------------------------------------
# Drawing + evidence
# ---------------------------------------------------------------------------
def _drawing_style(image: np.ndarray) -> tuple[float, int]:
    """Scale font size and line thickness with the image so labels stay readable."""
    longest_side = max(image.shape[:2])
    font_scale = max(0.45, min(1.1, longest_side / 1600))
    thickness = max(2, int(round(longest_side / 640)))
    return font_scale, thickness


def _draw_tag(
    image: np.ndarray, text: str, left: int, top: int, color: tuple[int, int, int],
    font_scale: float, text_thickness: int, tag_height: int,
) -> None:
    """Filled label rectangle with white text."""
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, text_thickness)
    cv2.rectangle(image, (left, top), (left + text_w + 2 * TAG_PADDING, top + tag_height), color, -1)
    cv2.putText(
        image, text, (left + TAG_PADDING, top + TAG_PADDING + text_h),
        FONT, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
    )


def _rects_overlap(a: list[int], b: list[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_block(
    image: np.ndarray, left: int, top: int, width: int, height: int, placed: list[list[int]] | None,
) -> tuple[int, int]:
    """
    Keep a label block inside the image and, when `placed` is given, shift it up
    or down in label-height steps so it does not cover labels drawn earlier.
    """
    img_h, img_w = image.shape[:2]
    left = max(0, min(left, img_w - width))
    top = max(0, min(top, img_h - height))
    if placed is None:
        return left, top
    for step in (0, -1, -2, -3, 1, 2, 3, -4, 4):
        candidate = top + step * height
        if candidate < 0 or candidate + height > img_h:
            continue
        rect = [left, candidate, left + width, candidate + height]
        if not any(_rects_overlap(rect, other) for other in placed):
            placed.append(rect)
            return left, candidate
    placed.append([left, top, left + width, top + height])
    return left, top


def draw_box_labels(
    image: np.ndarray,
    bbox: list[int],
    color: tuple[int, int, int],
    labels: list[str],
    font_scale: float,
    thickness: int,
    bottom_label: str | None = None,
    placed: list[list[int]] | None = None,
) -> None:
    """Stack labels above a box (inside it at the top edge) plus an optional label below it."""
    x1, y1, x2, y2 = bbox
    text_thickness = max(1, thickness - 1)
    (_, text_h), baseline = cv2.getTextSize("Ag", FONT, font_scale, text_thickness)
    tag_height = text_h + baseline + 2 * TAG_PADDING

    if labels:
        widths = [cv2.getTextSize(label, FONT, font_scale, text_thickness)[0][0] + 2 * TAG_PADDING for label in labels]
        block_height = tag_height * len(labels)
        preferred_top = y1 - block_height if y1 - block_height >= 0 else y1
        left, top = _place_block(image, x1, preferred_top, max(widths), block_height, placed)
        for label in labels:
            _draw_tag(image, label, left, top, color, font_scale, text_thickness, tag_height)
            top += tag_height

    if bottom_label:
        width = cv2.getTextSize(bottom_label, FONT, font_scale, text_thickness)[0][0] + 2 * TAG_PADDING
        preferred_top = y2 if y2 + tag_height <= image.shape[0] - 1 else y2 - tag_height
        left, top = _place_block(image, x1, preferred_top, width, tag_height, placed)
        _draw_tag(image, bottom_label, left, top, color, font_scale, text_thickness, tag_height)


def draw_box(
    image: np.ndarray,
    bbox: list[int],
    color: tuple[int, int, int],
    labels: list[str],
    font_scale: float,
    thickness: int,
    bottom_label: str | None = None,
) -> None:
    """Draw one box with its labels (used for single boxes, e.g. by the seed script)."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    draw_box_labels(image, bbox, color, labels, font_scale, thickness, bottom_label)


def annotate_frame(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
    """
    Draw every detection. Violating motorcycles and their violating riders are red
    with labels such as "NO HELMET 92%" / "TRIPLE RIDING 88%"; everything else is green.
    """
    annotated = frame.copy()
    font_scale, thickness = _drawing_style(annotated)
    red, green = config.COLOR_VIOLATION, config.COLOR_NORMAL

    # Default style for every detection: green box with "class confidence".
    styles: dict[int, dict[str, Any]] = {
        i: {"color": green, "labels": [f"{det['class']} {det['confidence']:.0%}"], "bottom": None}
        for i, det in enumerate(analysis.detections)
    }
    head_boxes: list[tuple[list[int], tuple[int, int, int]]] = []

    for group in analysis.motorcycles:
        types = {v["type"] for v in group.violations}
        if group.violations:
            labels = [f"{v['type'].replace('_', ' ')} {v['confidence']:.0%}" for v in group.violations]
        else:
            labels = [f"motorcycle {group.confidence:.0%}"]
        plate = group.plate["plate_number"] if group.plate else None
        styles[group.index] = {
            "color": red if group.violations else green,
            "labels": labels,
            "bottom": plate if plate and plate != ocr.NOT_DETECTED else None,
        }

        for rider in group.riders:
            violating = VIOLATION_TRIPLE_RIDING in types or (
                VIOLATION_NO_HELMET in types and rider.helmet_status == NO_HELMET
            )
            if rider.helmet_status == NO_HELMET:
                label = f"NO HELMET {rider.helmet_confidence:.0%}"
            elif rider.helmet_status == HELMET:
                label = f"helmet {rider.helmet_confidence:.0%}"
            else:
                label = f"rider {rider.confidence:.0%}"
            styles[rider.person_index] = {"color": red if violating else green, "labels": [label], "bottom": None}
            if rider.helmet_bbox is not None:
                head_boxes.append((rider.helmet_bbox, red if rider.helmet_status == NO_HELMET else green))

    # Thin boxes around matched helmet / bare-head detections.
    for bbox, color in head_boxes:
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, max(1, thickness - 1), cv2.LINE_AA)

    # Pass 1: box outlines, green first and red last so violations sit on top.
    for index in sorted(styles, key=lambda i: styles[i]["color"] == red):
        x1, y1, x2, y2 = analysis.detections[index]["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), styles[index]["color"], thickness, cv2.LINE_AA)

    # Pass 2: labels on top of every line. Violation labels are placed first so
    # they get their preferred spot; overlapping labels are shifted apart.
    placed: list[list[int]] = []
    for index in sorted(styles, key=lambda i: styles[i]["color"] != red):
        style = styles[index]
        draw_box_labels(
            annotated, analysis.detections[index]["bbox"], style["color"], style["labels"],
            font_scale, thickness, bottom_label=style["bottom"], placed=placed,
        )
    return annotated


def add_evidence_footer(image: np.ndarray, text: str) -> np.ndarray:
    """Append a dark strip with a timestamp caption below the image."""
    width = image.shape[1]
    font_scale = max(0.45, min(0.8, width / 1800))
    (_, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    bar_height = text_h + baseline + 16
    footer = np.full((bar_height, width, 3), 18, dtype=np.uint8)
    cv2.putText(footer, text, (10, 8 + text_h), FONT, font_scale, (225, 225, 225), 1, cv2.LINE_AA)
    return np.vstack([image, footer])


def save_evidence(image: np.ndarray, prefix: str = "evidence") -> str:
    """Save an annotated image to evidence/ with a unique name and return the file name."""
    filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:10]}.jpg"
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, config.EVIDENCE_JPEG_QUALITY])
    if not ok:
        raise DetectionError("Could not encode the evidence image")
    try:
        # imencode + tofile also works for non-ASCII Windows paths (cv2.imwrite does not).
        buffer.tofile(str(config.EVIDENCE_DIR / filename))
    except OSError as exc:
        logger.exception("Could not write evidence file %s", filename)
        raise DetectionError(f"Could not save the evidence image: {exc}") from exc
    return filename


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------
class VideoError(RuntimeError):
    """The video could not be opened or read."""


class VideoTooLongError(VideoError):
    """The video is longer than MAX_VIDEO_DURATION_SECONDS."""


@dataclass
class ViolationTrack:
    """One unique violation (one type, one vehicle) followed across sampled frames."""
    type: str
    bbox: list[int]
    last_frame: int
    hits: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)   # best frames, best first


def _same_vehicle(a: list[int], b: list[int]) -> float:
    """Match strength between two motorcycle boxes in nearby frames (0 = different vehicle)."""
    overlap = iou(a, b)
    if overlap >= config.VIDEO_MERGE_IOU:
        return 1.0 + overlap
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    diagonal = max(1.0, ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return 1.0 - distance / diagonal if distance <= config.VIDEO_MERGE_CENTER_RATIO * diagonal else 0.0


def _add_candidate(
    tracks: list[ViolationTrack], violation: dict[str, Any], group: MotorcycleGroup,
    frame_index: int, frame: np.ndarray, raw: np.ndarray, analysis: FrameAnalysis, max_gap: int,
) -> None:
    """Attach a per-frame violation to the best matching track, or start a new track."""
    best_track, best_strength = None, 0.0
    for track in tracks:
        if track.type != violation["type"] or frame_index - track.last_frame > max_gap:
            continue
        strength = _same_vehicle(track.bbox, group.bbox)
        if strength > best_strength:
            best_track, best_strength = track, strength
    if best_track is None:
        best_track = ViolationTrack(type=violation["type"], bbox=group.bbox, last_frame=frame_index)
        tracks.append(best_track)

    best_track.bbox, best_track.last_frame = group.bbox, frame_index
    best_track.hits += 1
    # Evidence quality = violation confidence + a little for a confident vehicle detection.
    best_track.candidates.append(
        {
            "score": violation["confidence"] + 0.1 * group.confidence,
            "frame_index": frame_index, "frame": frame, "raw": raw,
            "analysis": analysis, "group": group, "violation": violation,
        }
    )
    best_track.candidates.sort(key=lambda c: c["score"], reverse=True)
    del best_track.candidates[config.VIDEO_CANDIDATES_PER_TRACK:]


def _pick_evidence(track: ViolationTrack) -> dict[str, Any]:
    """Read plates on the best frames and keep the frame with the best combined score."""
    for position, candidate in enumerate(track.candidates):
        candidate["plate"] = {"plate_number": ocr.NOT_DETECTED, "confidence": 0.0, "raw_text": ""}
        if models.ocr_reader is not None and position < config.VIDEO_OCR_ATTEMPTS:
            raw, frame = candidate["raw"], candidate["frame"]
            scale = raw.shape[1] / frame.shape[1]
            bbox = clip_bbox([v * scale for v in candidate["group"].bbox], raw.shape[1], raw.shape[0])
            with models.lock:
                candidate["plate"] = ocr.read_plate(models.ocr_reader, raw, bbox, "motorcycle")
        if candidate["plate"]["plate_number"] != ocr.NOT_DETECTED:
            candidate["score"] += 0.25                 # a readable plate makes better evidence
    return max(track.candidates, key=lambda c: c["score"])


def process_video(path: Any, frame_interval: int, source_name: str) -> dict[str, Any]:
    """
    Analyse every `frame_interval`-th frame, merge repeated sightings of the same
    violation, and save one evidence image per unique violation.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError("The video could not be opened. Is it a valid MP4 file?")

    tracks: list[ViolationTrack] = []
    frames_processed = candidates = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise VideoError("Could not read the video's frame rate or length.")
        duration = frame_count / fps
        if duration > config.MAX_VIDEO_DURATION_SECONDS + 0.5:
            raise VideoTooLongError(
                f"Video is {duration:.1f}s long; the limit is {config.MAX_VIDEO_DURATION_SECONDS}s."
            )
        max_gap = max(frame_interval, int(round(fps * config.VIDEO_MERGE_MAX_GAP_SECONDS)))

        frame_index = -1
        while True:
            frame_index += 1
            if frame_index % frame_interval:
                if not capture.grab():                 # skip without decoding (fast)
                    break
                continue
            ok, raw = capture.read()
            if not ok or raw is None:
                break
            frames_processed += 1
            frame = resize_for_inference(raw)
            analysis = analyze_frame(frame, read_plates=False)   # OCR later, only on the best frames
            for group in analysis.motorcycles:
                for violation in group.violations:
                    candidates += 1
                    _add_candidate(tracks, violation, group, frame_index, frame, raw, analysis, max_gap)
    finally:
        capture.release()

    if frames_processed == 0:
        raise VideoError("No frames could be decoded from the video.")

    # Pick the best evidence frame per track, then merge tracks that share a readable plate.
    chosen: dict[tuple[str, str], tuple[ViolationTrack, dict[str, Any]]] = {}
    unplated: list[tuple[ViolationTrack, dict[str, Any]]] = []
    for track in tracks:
        best = _pick_evidence(track)
        plate = best["plate"]["plate_number"]
        if plate == ocr.NOT_DETECTED:
            unplated.append((track, best))
            continue
        key = (track.type, plate)
        if key in chosen:
            other_track, other_best = chosen[key]
            other_track.hits += track.hits
            if best["score"] > other_best["score"]:
                chosen[key] = (other_track, best)
        else:
            chosen[key] = (track, best)

    unique = []
    for track, best in sorted([*chosen.values(), *unplated], key=lambda item: item[1]["frame_index"]):
        group, analysis = best["group"], best["analysis"]
        group.plate = best["plate"]
        for violation in group.violations:
            violation["plate_number"] = best["plate"]["plate_number"]
            violation["plate_confidence"] = best["plate"]["confidence"]
        seconds = best["frame_index"] / fps
        footer = (
            f"TrafficEye evidence | {source_name} @ {seconds:.1f}s (frame {best['frame_index']}) | "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | helmet check: {analysis.helmet_check}"
        )
        evidence_name = save_evidence(add_evidence_footer(annotate_frame(best["frame"], analysis), footer), prefix="video")
        unique.append(
            {
                **best["violation"],
                "frame": best["frame_index"],
                "time_seconds": round(seconds, 2),
                "merged_detections": track.hits,
                "evidence_url": f"/evidence/{evidence_name}",
                "source_label": f"{source_name} @ {seconds:.1f}s",
            }
        )

    logger.info(
        "Video %s: %d frames analysed, %d violation sightings merged into %d unique violations",
        source_name, frames_processed, candidates, len(unique),
    )
    return {
        "frames_processed": frames_processed,
        "frame_interval": frame_interval,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "candidates_found": candidates,
        "violations": unique,
        "helmet_model_loaded": models.helmet_loaded,
    }


# ---------------------------------------------------------------------------
# Full image pipeline
# ---------------------------------------------------------------------------
def process_image(image: np.ndarray) -> dict[str, Any]:
    """
    Analyse one uploaded image and save its evidence image.
    Runs in a worker thread (see main.py), so blocking work is fine here.
    """
    frame = resize_for_inference(image)
    height, width = frame.shape[:2]
    scale_to_original = image.shape[1] / width

    analysis = analyze_frame(frame, original=image, scale_to_original=scale_to_original)

    annotated = annotate_frame(frame, analysis)
    footer = (
        f"TrafficEye evidence | {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"{len(analysis.detections)} objects | {len(analysis.violations)} violations | "
        f"helmet check: {analysis.helmet_check}"
    )
    evidence_name = save_evidence(add_evidence_footer(annotated, footer))

    return {
        "image_width": width,
        "image_height": height,
        **analysis.to_response(),
        "evidence_url": f"/evidence/{evidence_name}",
    }
