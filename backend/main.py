"""
TrafficEye – FastAPI application entry point.

Run from the project root (the folder that contains backend/ and frontend/):

    uvicorn backend.main:app --reload

Then open http://localhost:8000 for the UI or http://localhost:8000/docs for Swagger.
"""
from __future__ import annotations

import io
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from http import HTTPStatus
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import config, database, detector, report
from backend.database import ViolationRecord, session_scope
from backend.detector import DetectionError, ModelNotLoadedError, models
from backend.report import ReportError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trafficeye.api")

# uploads/, evidence/ and reports/ must exist before StaticFiles mounts them below.
config.ensure_directories()


# ===========================================================================
# API schemas (these also drive the Swagger documentation at /docs)
# ===========================================================================
class ErrorResponse(BaseModel):
    success: bool = False
    error: str = Field(examples=["Unsupported file type"])
    detail: str = Field(default="", examples=["Only JPG and PNG images are allowed."])


class HealthResponse(BaseModel):
    success: bool = True
    yolo_loaded: bool
    helmet_model_loaded: bool
    ocr_loaded: bool
    device: str = Field(examples=["cpu"])
    helmet_classes: list[str] = Field(default_factory=list)
    helmet_class_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="How each helmet.pt class is interpreted: HELMET, NO_HELMET or IGNORED",
        examples=[{"With Helmet": "HELMET", "Without Helmet": "NO_HELMET"}],
    )
    errors: dict[str, str] = Field(default_factory=dict)


class Detection(BaseModel):
    # "class" is a Python keyword, so the field is named class_name and exposed as "class".
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class", examples=["motorcycle"])
    confidence: float = Field(examples=[0.91])
    bbox: list[int] = Field(examples=[[120, 80, 460, 520]], description="[x1, y1, x2, y2] in pixels")


class Rider(BaseModel):
    bbox: list[int]
    confidence: float
    helmet_status: str = Field(description="HELMET, NO_HELMET or UNKNOWN", examples=["NO_HELMET"])
    helmet_confidence: float | None = Field(default=None, examples=[0.92])


class Motorcycle(BaseModel):
    bbox: list[int]
    confidence: float
    rider_count: int = Field(examples=[3])
    riders: list[Rider]
    plate_number: str | None = Field(
        default=None,
        description="Read only for violating motorcycles; null when OCR was not attempted",
        examples=["TN 33 AB 1234"],
    )
    violations: list[str] = Field(examples=[["TRIPLE_RIDING", "NO_HELMET"]])


class Violation(BaseModel):
    type: str = Field(examples=["NO_HELMET"])
    confidence: float = Field(examples=[0.92])
    fine: int = Field(examples=[1000])
    plate_number: str = Field(examples=["TN 33 AB 1234"])
    plate_confidence: float = Field(examples=[0.81])
    vehicle_class: str = Field(examples=["motorcycle"])
    vehicle_bbox: list[int]
    rider_count: int
    riders_without_helmet: int
    rider_bboxes: list[list[int]]
    description: str = Field(examples=["1 of 2 rider(s) without a helmet"])
    # Filled in once the violation is saved to the database.
    id: int | None = Field(default=None, examples=[1])
    challan_id: str | None = Field(default=None, examples=["TE-20260925-00001"])
    evidence_url: str | None = Field(default=None, examples=["/evidence/evidence_20260925_101500_1a2b3c4d5e.jpg"])
    report_url: str | None = Field(default=None, examples=["/api/report/1"])
    created_at: str | None = Field(default=None, examples=["2026-09-25T10:15:00"])
    # Video only: where the best evidence frame is and how many sightings were merged.
    frame: int | None = Field(default=None, examples=[120])
    time_seconds: float | None = Field(default=None, examples=[4.0])
    merged_detections: int | None = Field(default=None, examples=[9])


class VideoDetectionResponse(BaseModel):
    success: bool = True
    filename: str
    frames_processed: int = Field(examples=[90])
    frame_interval: int = Field(examples=[10])
    fps: float = Field(examples=[30.0])
    duration_seconds: float = Field(examples=[30.0])
    candidates_found: int = Field(description="Per-frame violation sightings before merging", examples=[27])
    violations: list[Violation] = Field(description="Unique violations after duplicate merging")
    helmet_model_loaded: bool
    processing_time_ms: int


class ImageDetectionResponse(BaseModel):
    success: bool = True
    filename: str
    image_width: int
    image_height: int
    detections: list[Detection]
    object_counts: dict[str, int]
    motorcycles: list[Motorcycle]
    violations: list[Violation]
    helmet_model_loaded: bool
    helmet_check: str = Field(description="enabled, skipped (no helmet.pt) or failed", examples=["skipped"])
    ocr_loaded: bool
    evidence_url: str = Field(examples=["/evidence/evidence_20260925_101500_1a2b3c4d5e.jpg"])
    processing_time_ms: int


class ViolationRecordOut(BaseModel):
    """One row of the violations table, as returned by the history API."""

    id: int = Field(examples=[1])
    challan_id: str = Field(examples=["TE-20260925-00001"])
    type: str = Field(examples=["NO_HELMET"])
    confidence: float = Field(examples=[0.92])
    plate_number: str = Field(examples=["TN 33 AB 1234"])
    fine: int = Field(examples=[1000])
    evidence_path: str = Field(examples=["evidence/evidence_20260925_101500_1a2b3c4d5e.jpg"])
    evidence_url: str = Field(examples=["/evidence/evidence_20260925_101500_1a2b3c4d5e.jpg"])
    report_path: str | None = Field(default=None, examples=["reports/challan_1.pdf"])
    report_url: str = Field(examples=["/api/report/1"])
    source_file: str = Field(examples=["street.jpg"])
    created_at: str = Field(examples=["2026-09-25T10:15:00"])


class ViolationListResponse(BaseModel):
    success: bool = True
    count: int
    filters: dict[str, str | None]
    violations: list[ViolationRecordOut]


class DailyCount(BaseModel):
    date: str = Field(examples=["2026-09-25"])
    count: int = Field(examples=[4])
    no_helmet: int = Field(examples=[3])
    triple_riding: int = Field(examples=[1])


class StatsResponse(BaseModel):
    success: bool = True
    total: int = Field(examples=[20])
    no_helmet: int = Field(examples=[13])
    triple_riding: int = Field(examples=[7])
    total_fines: int = Field(examples=[20000])
    today: int = Field(examples=[4])
    by_type: dict[str, int] = Field(examples=[{"NO_HELMET": 13, "TRIPLE_RIDING": 7}])
    daily_trend: list[DailyCount] = Field(description="One entry per day, oldest first; empty days are 0")
    recent: list[ViolationRecordOut]


class DeleteResponse(BaseModel):
    success: bool = True
    deleted_id: int
    report_deleted: bool
    evidence_deleted: bool = Field(description="False when other violations still use the same evidence image")


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid or unsupported upload"},
    413: {"model": ErrorResponse, "description": "File too large"},
    422: {"model": ErrorResponse, "description": "Malformed request"},
    500: {"model": ErrorResponse, "description": "Detection failed"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


# ===========================================================================
# Error handling – every error uses {"success": false, "error": ..., "detail": ...}
# ===========================================================================
class APIError(Exception):
    """Raise anywhere in a request to return a clean JSON error."""

    def __init__(self, status_code: int, error: str, detail: str = "") -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def error_json(status_code: int, error: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": error, "detail": detail},
    )


# ===========================================================================
# Lifespan – load models ONCE at startup, never per request
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_directories()
    database.init_db()
    logger.info("Starting TrafficEye - loading models (the first run downloads weights)...")
    await run_in_threadpool(detector.load_models)
    logger.info("TrafficEye ready - UI at / and API docs at /docs")
    yield
    detector.unload_models()
    logger.info("TrafficEye stopped")


app = FastAPI(
    title="TrafficEye API",
    description=(
        "Intelligent Traffic Violation Detection System. "
        "Upload traffic images to detect vehicles and riders with YOLOv8, "
        "flag triple riding and riding without a helmet, read number plates "
        "with EasyOCR, store violations in SQLite and download PDF e-challans."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("%s %s -> %s %s: %s", request.method, request.url.path, exc.status_code, exc.error, exc.detail)
    return error_json(exc.status_code, exc.error, exc.detail)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "HTTP error"
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if exc.status_code == 404 and detail == phrase:
        detail = f"Nothing found at {request.url.path}"
    return error_json(exc.status_code, phrase, detail)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()) if part not in ("body", "query", "path"))
        message = err.get("msg", "Invalid value")
        messages.append(f"{location}: {message}" if location else message)
    return error_json(422, "Invalid request", "; ".join(messages))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_json(
        500,
        "Internal server error",
        "Something went wrong while processing the request. Check the server logs.",
    )


# ===========================================================================
# Middleware
# ===========================================================================
# Per-endpoint upload limits, checked from Content-Length before the body is read.
UPLOAD_LIMITS = {
    "/api/detect/image": (config.MAX_IMAGE_SIZE, f"Maximum image size is {config.MAX_IMAGE_SIZE_MB} MB."),
    "/api/detect/video": (config.MAX_VIDEO_SIZE, f"Maximum video size is {config.MAX_VIDEO_SIZE_MB} MB."),
}
MULTIPART_OVERHEAD = 64 * 1024          # room for multipart boundaries and headers


@app.middleware("http")
async def upload_guard_and_cache_headers(request: Request, call_next):
    """
    1. Reject oversized uploads early using the Content-Length header.
    2. Ask browsers to revalidate HTML/CSS/JS so UI updates show up immediately.
    """
    limit = UPLOAD_LIMITS.get(request.url.path)
    content_length = request.headers.get("content-length", "")
    if request.method == "POST" and limit and content_length.isdigit():
        max_bytes, message = limit
        if int(content_length) > max_bytes + MULTIPART_OVERHEAD:
            logger.warning("Rejected %s upload of %s bytes", request.url.path, content_length)
            return error_json(413, "File too large", message)

    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/dashboard", "/history") or path.endswith(".html") or path.startswith(("/css/", "/js/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


# Added last so it is the outermost middleware and also covers early 413 responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Upload validation helpers
# ===========================================================================
UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._ ()-]+")


def sanitize_filename(filename: str | None) -> str:
    """
    Make a safe *display* name from the client-supplied file name.
    It is only shown to users / stored as text – files on disk always get UUID names.
    """
    name = re.split(r"[\\/]", filename or "")[-1]          # drop any directory part
    name = UNSAFE_NAME_CHARS.sub("_", name).strip(" .")
    return name[:120] or "upload"


def validate_extension(filename: str, allowed: set[str], message: str) -> str:
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if extension not in allowed:
        raise APIError(400, "Unsupported file type", message)
    return extension


async def read_upload(upload: UploadFile, max_bytes: int, max_mb: int) -> bytes:
    """Read an upload in chunks and stop as soon as it exceeds the size limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(config.UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise APIError(413, "File too large", f"Maximum allowed size is {max_mb} MB.")
        chunks.append(chunk)
    if total == 0:
        raise APIError(400, "Empty file", "The uploaded file is empty.")
    return b"".join(chunks)


def decode_image(data: bytes) -> np.ndarray:
    """
    Verify the bytes really are a JPG/PNG (content check, not just the extension)
    and decode them into an OpenCV BGR image.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            width, height = probe.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise APIError(400, "Invalid image", "The file content is not a readable JPG or PNG image.") from None

    if image_format not in ("JPEG", "PNG"):
        raise APIError(400, "Unsupported file type", f"File content is {image_format}; only JPG and PNG are allowed.")
    if width * height > config.MAX_IMAGE_PIXELS:
        raise APIError(400, "Image too large", f"Image is {width}x{height}; the limit is {config.MAX_IMAGE_PIXELS // 1_000_000} megapixels.")
    if min(width, height) < config.MIN_IMAGE_SIDE:
        raise APIError(400, "Image too small", f"Image is {width}x{height}; each side must be at least {config.MIN_IMAGE_SIDE} px.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise APIError(400, "Invalid image", "The image could not be decoded. It may be corrupted.")
    return image


def save_upload(data: bytes, extension: str) -> str | None:
    """Keep a copy of the original upload under a random UUID name (never the client name)."""
    stored_name = f"{uuid.uuid4().hex}{extension}"
    try:
        (config.UPLOAD_DIR / stored_name).write_bytes(data)
        return stored_name
    except OSError:
        logger.exception("Could not store the original upload")
        return None


# ===========================================================================
# Violation storage + challans
# ===========================================================================
def build_challan(record: ViolationRecord) -> str | None:
    """
    Generate a record's PDF. A failure never fails the detection request: it is
    logged and the PDF is generated again when the challan is first downloaded.
    """
    try:
        return report.generate_challan(record)
    except ReportError:
        return None                                   # already logged by report.py
    except Exception:
        logger.exception("Unexpected error generating the challan for violation %s", record.id)
        return None


def persist_violations(violations: list[dict[str, Any]], source_file: str, evidence_url: str | None = None) -> None:
    """
    Save violations to SQLite and generate their PDF e-challans.
    Image violations all share the image's one evidence file (`evidence_url`);
    video violations each carry their own "evidence_url" and "source_label".
    Runs in a worker thread; adds id / challan_id / report_url to each violation.
    """
    if not violations:
        return

    detected_at = datetime.now()
    with session_scope() as session:
        records = [
            ViolationRecord(
                type=v["type"],
                confidence=v["confidence"],
                plate_number=v["plate_number"],
                fine=v["fine"],
                evidence_path=database.stored_path("evidence", (v.get("evidence_url") or evidence_url).rsplit("/", 1)[-1]),
                source_file=v.get("source_label") or source_file,
                created_at=detected_at,
            )
            for v in violations
        ]
        session.add_all(records)
        session.flush()                                     # assigns the ids
        for record in records:
            record.report_path = build_challan(record)

    for violation, record in zip(violations, records):
        saved = record.to_dict()
        for key in ("id", "challan_id", "evidence_url", "report_url", "created_at"):
            violation[key] = saved[key]


def remove_file(path: Any) -> bool:
    """Delete a file if it exists. Returns True if something was deleted."""
    if path is None or not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        logger.exception("Could not delete %s", path)
        return False


def parse_violation_type(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    violation_type = value.strip().upper()
    if violation_type not in config.FINES:
        raise APIError(400, "Invalid violation type", f"Use one of: {', '.join(config.FINES)}.")
    return violation_type


# ===========================================================================
# API endpoints
# ===========================================================================
@app.get("/api/health", response_model=HealthResponse, tags=["System"], summary="Model and server status")
async def health() -> dict[str, Any]:
    """Reports which models loaded at startup. The navbar status dots use this."""
    return {"success": True, **models.status()}


@app.post(
    "/api/detect/image",
    response_model=ImageDetectionResponse,
    responses=ERROR_RESPONSES,
    tags=["Detection"],
    summary="Detect vehicles, riders and violations in an image",
)
async def detect_image(
    file: UploadFile = File(..., description="Traffic image - JPG or PNG, max 10 MB"),
) -> dict[str, Any]:
    """
    Upload a traffic image. TrafficEye runs YOLOv8, links riders to motorcycles,
    checks helmets (when helmet.pt is loaded) and triple riding, reads the plate
    of every violating motorcycle, saves an annotated evidence image, stores each
    violation in SQLite and generates its PDF e-challan.
    """
    started = time.perf_counter()

    display_name = sanitize_filename(file.filename)
    extension = validate_extension(
        display_name, config.ALLOWED_IMAGE_EXTENSIONS, "Only JPG and PNG images are allowed."
    )
    data = await read_upload(file, config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE_MB)
    image = await run_in_threadpool(decode_image, data)

    if not models.yolo_loaded:
        raise APIError(
            503,
            "Detection model unavailable",
            models.errors.get("yolo", "YOLOv8 failed to load at startup. Check the server logs."),
        )

    save_upload(data, extension)

    try:
        result = await run_in_threadpool(detector.process_image, image)
    except ModelNotLoadedError as exc:
        raise APIError(503, "Detection model unavailable", str(exc)) from exc
    except DetectionError as exc:
        raise APIError(500, "Detection failed", str(exc)) from exc

    try:
        await run_in_threadpool(persist_violations, result["violations"], display_name, result["evidence_url"])
    except SQLAlchemyError as exc:
        logger.exception("Could not save violations for %s", display_name)
        raise APIError(500, "Database error", "Violations were detected but could not be saved. Check the server logs.") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Analysed %s: %d objects, %d violations in %d ms",
        display_name, len(result["detections"]), len(result["violations"]), elapsed_ms,
    )
    return {"success": True, "filename": display_name, **result, "processing_time_ms": elapsed_ms}


@app.post(
    "/api/detect/video",
    response_model=VideoDetectionResponse,
    responses=ERROR_RESPONSES,
    tags=["Detection"],
    summary="Detect violations in an MP4 video (duplicates merged)",
)
async def detect_video(
    file: UploadFile = File(..., description="MP4 video, max 50 MB and 30 seconds"),
    frame_interval: int = Query(config.FRAME_INTERVAL, ge=1, le=120, description="Analyse every Nth frame"),
) -> dict[str, Any]:
    """
    Samples every Nth frame, runs the same detection + violation rules as images,
    merges repeated sightings of the same violation (same type, overlapping bike
    box, close in time, or same plate) and saves ONE record per unique violation
    with its best evidence frame.
    """
    started = time.perf_counter()
    display_name = sanitize_filename(file.filename)
    validate_extension(display_name, config.ALLOWED_VIDEO_EXTENSIONS, "Only MP4 videos are allowed.")
    data = await read_upload(file, config.MAX_VIDEO_SIZE, config.MAX_VIDEO_SIZE_MB)
    if data[4:8] != b"ftyp":                                  # every MP4 starts with an 'ftyp' box
        raise APIError(400, "Invalid video", "The file content is not an MP4 video.")
    if not models.yolo_loaded:
        raise APIError(503, "Detection model unavailable", models.errors.get("yolo", "YOLOv8 failed to load."))

    stored_name = save_upload(data, ".mp4")
    if stored_name is None:
        raise APIError(500, "Upload failed", "The video could not be saved for processing.")
    video_path = config.UPLOAD_DIR / stored_name

    try:
        result = await run_in_threadpool(detector.process_video, video_path, frame_interval, display_name)
    except detector.VideoTooLongError as exc:
        video_path.unlink(missing_ok=True)
        raise APIError(400, "Video too long", str(exc)) from exc
    except detector.VideoError as exc:
        video_path.unlink(missing_ok=True)
        raise APIError(400, "Invalid video", str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise APIError(503, "Detection model unavailable", str(exc)) from exc
    except DetectionError as exc:
        raise APIError(500, "Detection failed", str(exc)) from exc

    try:
        await run_in_threadpool(persist_violations, result["violations"], display_name)
    except SQLAlchemyError as exc:
        logger.exception("Could not save video violations for %s", display_name)
        raise APIError(500, "Database error", "Violations were detected but could not be saved.") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {"success": True, "filename": display_name, **result, "processing_time_ms": elapsed_ms}


@app.get("/api/stats", response_model=StatsResponse, tags=["Violations"], summary="Dashboard statistics")
def stats(days: int = Query(7, ge=1, le=90, description="Length of the daily trend in days")) -> dict[str, Any]:
    """Totals per violation type, total fines, and a daily trend with missing days filled with 0."""
    with session_scope() as session:
        return {"success": True, **database.get_stats(session, days)}


@app.get(
    "/api/violations",
    response_model=ViolationListResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["Violations"],
    summary="List saved violations (filters can be combined)",
)
def list_violations(
    violation_type: str | None = Query(None, alias="type", description="NO_HELMET or TRIPLE_RIDING"),
    day: date | None = Query(None, alias="date", description="Only violations on this day, YYYY-MM-DD"),
    search: str | None = Query(None, max_length=32, description="Plate search; spaces and case are ignored"),
    limit: int = Query(config.MAX_HISTORY_RESULTS, ge=1, le=config.MAX_HISTORY_RESULTS),
) -> dict[str, Any]:
    """Examples: `?type=NO_HELMET`, `?date=2026-09-25`, `?search=TN33`, `?type=TRIPLE_RIDING&date=2026-09-25`."""
    parsed_type = parse_violation_type(violation_type)
    search_text = search.strip() if search and search.strip() else None
    with session_scope() as session:
        rows = [record.to_dict() for record in database.list_violations(session, parsed_type, day, search_text, limit)]
    return {
        "success": True,
        "count": len(rows),
        "filters": {"type": parsed_type, "date": day.isoformat() if day else None, "search": search_text},
        "violations": rows,
    }


@app.get(
    "/api/report/{violation_id}",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/pdf": {}}, "description": "The PDF e-challan"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Violations"],
    summary="Download the PDF e-challan for a violation",
)
def download_report(
    violation_id: int = PathParam(..., ge=1),
    download: bool = Query(True, description="false opens the PDF in the browser instead of downloading it"),
) -> FileResponse:
    """Returns challan_<id>.pdf. If the file is missing it is regenerated first."""
    with session_scope() as session:
        record = session.get(ViolationRecord, violation_id)
        if record is None:
            raise APIError(404, "Violation not found", f"No violation with id {violation_id}.")
        pdf_path = database.resolve_stored_path(record.report_path, config.REPORT_DIR)
        if pdf_path is None or not pdf_path.is_file():
            logger.info("Challan for violation %s missing - regenerating", violation_id)
            try:
                record.report_path = report.generate_challan(record)
            except ReportError as exc:
                raise APIError(500, "Challan generation failed", str(exc)) from exc
            pdf_path = database.resolve_stored_path(record.report_path, config.REPORT_DIR)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"challan_{violation_id}.pdf",
        content_disposition_type="attachment" if download else "inline",
    )


@app.delete(
    "/api/violations/{violation_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Violations"],
    summary="Delete a violation, its challan and (if unused) its evidence image",
)
def delete_violation(violation_id: int = PathParam(..., ge=1)) -> dict[str, Any]:
    """
    One image can produce several violations that share one evidence file, so the
    evidence image is deleted only when no other violation still references it.
    """
    with session_scope() as session:
        record = session.get(ViolationRecord, violation_id)
        if record is None:
            raise APIError(404, "Violation not found", f"No violation with id {violation_id}.")
        evidence_path, report_path = record.evidence_path, record.report_path
        session.delete(record)
        session.flush()
        remaining_references = database.count_evidence_references(session, evidence_path)

    # Files are removed only after the database change has been committed.
    report_files = {database.resolve_stored_path(report_path, config.REPORT_DIR), report.challan_path(violation_id)}
    report_deleted = any([remove_file(path) for path in report_files])
    evidence_deleted = remaining_references == 0 and remove_file(
        database.resolve_stored_path(evidence_path, config.EVIDENCE_DIR)
    )
    logger.info(
        "Deleted violation %s (report deleted: %s, evidence deleted: %s, evidence still used by %d)",
        violation_id, report_deleted, evidence_deleted, remaining_references,
    )
    return {
        "success": True,
        "deleted_id": violation_id,
        "report_deleted": report_deleted,
        "evidence_deleted": evidence_deleted,
    }


# ===========================================================================
# Frontend + static files
# ===========================================================================
def serve_page(filename: str) -> FileResponse:
    path = config.FRONTEND_DIR / filename
    if not path.is_file():
        raise APIError(404, "Page not found", f"frontend/{filename} is missing.")
    return FileResponse(path, media_type="text/html")


@app.get("/", include_in_schema=False)
async def home_page() -> FileResponse:
    return serve_page("index.html")


@app.get("/index.html", include_in_schema=False)
async def home_page_alias() -> FileResponse:
    return serve_page("index.html")


@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard.html", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return serve_page("dashboard.html")


@app.get("/history", include_in_schema=False)
@app.get("/history.html", include_in_schema=False)
async def history_page() -> FileResponse:
    return serve_page("history.html")


# StaticFiles resolves paths safely, so "../" tricks cannot escape these folders.
app.mount("/css", StaticFiles(directory=config.FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=config.FRONTEND_DIR / "js"), name="js")
app.mount("/evidence", StaticFiles(directory=config.EVIDENCE_DIR), name="evidence")
