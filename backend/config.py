"""
TrafficEye – central configuration.

Every tunable number lives here so the team can adjust thresholds and limits
during a demo without hunting through the code. All paths are built with
pathlib relative to the project root, so the app runs the same on Windows
and Linux no matter where the folder is placed.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      # .../trafficeye
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BACKEND_DIR / "models"

UPLOAD_DIR = BASE_DIR / "uploads"       # original uploads (UUID file names)
EVIDENCE_DIR = BASE_DIR / "evidence"    # annotated evidence images (served at /evidence)
REPORT_DIR = BASE_DIR / "reports"       # generated PDF e-challans

# Folders the app creates automatically on startup if they are missing.
RUNTIME_DIRS = (UPLOAD_DIR, EVIDENCE_DIR, REPORT_DIR, MODELS_DIR)

# SQLite database file (created automatically on startup).
DATABASE_PATH = BASE_DIR / "trafficeye.db"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Ultralytics downloads yolov8n.pt to this path on the first run.
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
# Optional helmet / no-helmet model – the user places it here manually.
HELMET_MODEL_PATH = MODELS_DIR / "helmet.pt"
# EasyOCR downloads its detector + recogniser weights here on the first run.
OCR_MODEL_DIR = MODELS_DIR / "easyocr"
OCR_LANGUAGES = ["en"]

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MB = 1024 * 1024

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE = MAX_IMAGE_SIZE_MB * MB
MAX_IMAGE_PIXELS = 40_000_000           # rejects "decompression bomb" images (> 40 MP)
MIN_IMAGE_SIDE = 32                     # smaller images cannot contain useful detections

MAX_VIDEO_SIZE_MB = 50
MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * MB
MAX_VIDEO_DURATION_SECONDS = 30

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}

UPLOAD_CHUNK_SIZE = 1 * MB              # uploads are read in chunks to enforce the size limit

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# COCO classes TrafficEye cares about; everything else YOLO sees is ignored.
TARGET_CLASSES = ("person", "motorcycle", "car", "bus", "truck")

YOLO_CONFIDENCE = 0.35                  # minimum confidence for a YOLO detection
IOU_THRESHOLD = 0.45                    # non-max-suppression IoU threshold
YOLO_IMAGE_SIZE = 640                   # network input size used by YOLOv8
MAX_FRAME_WIDTH = 1280                  # images/frames wider than this are downscaled first
FRAME_INTERVAL = 10                     # video: analyse every Nth frame

# ---------------------------------------------------------------------------
# Video duplicate merging
# ---------------------------------------------------------------------------
# The same bike appears in many frames. Candidates of the same violation type
# are merged into one when their motorcycle boxes overlap (IoU) or their centres
# are close, and they are no more than VIDEO_MERGE_MAX_GAP_SECONDS apart.
# After OCR, tracks with the same readable plate are merged as well.
VIDEO_MERGE_IOU = 0.30
VIDEO_MERGE_CENTER_RATIO = 0.5          # centre distance <= 0.5 x box diagonal
VIDEO_MERGE_MAX_GAP_SECONDS = 2.0
VIDEO_CANDIDATES_PER_TRACK = 3          # best frames kept per unique violation
VIDEO_OCR_ATTEMPTS = 2                  # plate reads tried per unique violation

# ---------------------------------------------------------------------------
# Rider <-> motorcycle association
# ---------------------------------------------------------------------------
# A person counts as a rider of a motorcycle when EITHER
#   * at least MIN_RIDER_OVERLAP of the person box lies on the motorcycle box, OR
#   * the person's bottom-centre point is inside the motorcycle box expanded
#     by RIDER_BOX_EXPAND_X / _Y on every side,
# and the size/position sanity checks below pass.
MIN_RIDER_OVERLAP = 0.25
RIDER_BOX_EXPAND_X = 0.15               # widen the motorcycle box by 15% per side
RIDER_BOX_EXPAND_Y = 0.25               # and heighten it by 25% per side
MIN_RIDER_HEIGHT_RATIO = 0.5            # rider height must be 0.5x - 3.5x the motorcycle height
MAX_RIDER_HEIGHT_RATIO = 3.5
MAX_RIDERS_PER_MOTORCYCLE = 2           # more riders than this = TRIPLE_RIDING

# ---------------------------------------------------------------------------
# Helmet check
# ---------------------------------------------------------------------------
HELMET_CONFIDENCE = 0.40                # minimum confidence for a helmet/no-helmet detection
HEAD_REGION_RATIO = 0.30                # head = top 30% of a rider's person box
HEAD_MATCH_MIN_OVERLAP = 0.30           # share of a helmet box that must fall inside the head region

# Helmet models name their classes differently. Names are lower-cased and
# stripped of spaces, "-" and "_" and then looked up here
# ("With Helmet" -> "withhelmet", "no-helmet" -> "nohelmet").
# Any other class the model has (e.g. "person", "plate") is ignored.
HELMET_CLASS_KEYS = {"helmet", "withhelmet", "wearinghelmet", "helmeton", "hardhat"}
NO_HELMET_CLASS_KEYS = {"nohelmet", "withouthelmet", "head", "barehead", "helmetoff", "nohardhat"}

# ---------------------------------------------------------------------------
# Number plate OCR
# ---------------------------------------------------------------------------
PLATE_NOT_DETECTED = "Not detected"
# Where to look for the plate inside a vehicle box: (left, top, right, bottom)
# as fractions of the box width/height.
PLATE_SEARCH_REGIONS = {
    "motorcycle": (0.05, 0.30, 0.95, 1.00),
    "car": (0.15, 0.50, 0.85, 1.00),
    "bus": (0.15, 0.55, 0.85, 1.00),
    "truck": (0.15, 0.55, 0.85, 1.00),
}
OCR_MIN_VEHICLE_HEIGHT = 40             # px; smaller vehicles are too far away to read
OCR_TARGET_CROP_HEIGHT = 320            # small crops are upscaled towards this height
OCR_MAX_UPSCALE = 3.0
OCR_MAX_CROP_WIDTH = 960                # large crops are downscaled to this width (OCR time grows with size)
OCR_MIN_TEXT_CONFIDENCE = 0.10          # EasyOCR fragments below this are discarded
OCR_MAX_VEHICLES_PER_IMAGE = 5          # caps OCR time on crowded images

# ---------------------------------------------------------------------------
# Fines (Indian Rupees)
# ---------------------------------------------------------------------------
FINES = {
    "NO_HELMET": 1000,
    "TRIPLE_RIDING": 1000,
}

# Text printed on the e-challan for each violation type.
VIOLATION_LABELS = {
    "NO_HELMET": "No Helmet",
    "TRIPLE_RIDING": "Triple Riding",
}
VIOLATION_OFFENCES = {
    "NO_HELMET": ("Riding without protective headgear", "Section 194D, Motor Vehicles Act, 1988"),
    "TRIPLE_RIDING": ("More than one pillion rider on a motorcycle", "Section 194C, Motor Vehicles Act, 1988"),
}

# ---------------------------------------------------------------------------
# History API
# ---------------------------------------------------------------------------
MAX_HISTORY_RESULTS = 500               # most records GET /api/violations returns at once
DEMO_SOURCE_PREFIX = "demo_seed_"       # source_file prefix that marks seed-script records

# ---------------------------------------------------------------------------
# Evidence images
# ---------------------------------------------------------------------------
EVIDENCE_JPEG_QUALITY = 90
COLOR_NORMAL = (60, 200, 80)            # OpenCV uses BGR: green
COLOR_VIOLATION = (40, 40, 230)         # red


def ensure_directories() -> None:
    """Create uploads/, evidence/, reports/ and backend/models/ if they are missing."""
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
