# TrafficEye – Intelligent Traffic Violation Detection System

**Problem title:** AI-05: Intelligent Traffic Violation Detection

## Problem statement
> Develop an AI-based prototype that analyzes traffic images or video frames to identify possible violations such as helmetless riding, seat-belt violations, or excessive vehicle occupancy. The system should detect the vehicle/person and display the identified violation.

India records well over a lakh road deaths a year, and two-wheeler riders make up the largest share. Riding without a helmet and carrying more than one pillion ("triple riding") are among the most common offences, but enforcement still depends on officers watching roads or reviewing camera footage by hand. That doesn't scale to the number of junctions and cameras in an Indian city, so most violations go unrecorded.

## Proposed solution
TrafficEye automates the first stage of enforcement with computer vision:

1. **YOLOv8** finds people, motorcycles, cars, buses and trucks in a photo or video.
2. A **violation engine** links riders to motorcycles, checks each rider's head for a helmet and counts riders per bike.
3. **EasyOCR** reads the number plate of every violating motorcycle.
4. Each violation is stored in **SQLite** with an annotated **evidence image** and a **PDF e-challan**, ready for an officer to verify.

## Features
- Image (JPG/PNG, max 10 MB) and video (MP4, max 50 MB and 30 s) analysis
- Triple-riding detection and helmet / no-helmet detection (with an optional helmet model)
- Number plate OCR with Indian-format validation (`TN 33 AB 1234`, `22 BH 1234 AA`)
- Annotated evidence images: red for violations, green for everything else
- PDF e-challans with fine, legal section, plate, confidence and evidence
- Video duplicate merging: one record per unique violation, with the best evidence frame
- Dashboard: stat cards, violations by type and a 7-day trend
- History: plate search, type and date filters, evidence preview, challan download and delete
- Live model status in the navbar, toast notifications, and the same error format on every endpoint
- Swagger docs at `/docs`; one FastAPI server and no frontend build step

## Architecture
```text
                    ┌────────────────────┐
                    │     Browser        │
                    │ HTML + Tailwind    │
                    │ Vanilla JavaScript │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │      FastAPI       │
                    │      REST API      │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ YOLOv8    │   │ EasyOCR   │   │ SQLite    │
        │ Detection │   │ Plate OCR │   │ Database  │
        └─────┬─────┘   └───────────┘   └───────────┘
              │
              ▼
        ┌──────────────┐
        │ Violation    │
        │ Engine       │
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ Evidence +   │
        │ PDF Challan  │
        └──────────────┘
```

| File | Role |
|---|---|
| `backend/main.py` | FastAPI app: routes, upload validation, error handling, static files |
| `backend/detector.py` | Model loading (once, at startup), detection, violation engine, evidence, video pipeline |
| `backend/ocr.py` | Plate cropping, preprocessing, EasyOCR, Indian plate normalisation |
| `backend/database.py` | SQLAlchemy model, queries, dashboard statistics |
| `backend/report.py` | ReportLab PDF e-challans |
| `backend/seed.py` | Demo data generator |
| `backend/config.py` | Every threshold, limit, fine and path |

## How the violation logic works
- **Rider association.** A person counts as a rider of a motorcycle if at least 25% of their box overlaps it, or if the bottom-centre of their box falls inside the motorcycle box enlarged by 15% sideways and 25% vertically. Size and position checks reject pedestrians and people in the background. Each person is assigned to their single best-matching motorcycle, so nobody is counted on two bikes.
- **Triple riding.** More than 2 riders on one motorcycle. Confidence is the average of the motorcycle's and its riders' detection confidences.
- **Helmet check.** The helmet model runs on the whole image. Each helmet / no-helmet box is matched one-to-one to a rider's **head region**, the top 30% of the rider's box. A rider is therefore `HELMET`, `NO_HELMET` or `UNKNOWN`.
- **UNKNOWN is never a violation.** Only a positively detected bare head raises `NO_HELMET`. Without `helmet.pt` the check is skipped entirely, and the UI says so.
- **Plate OCR.** Runs on the lower part of each violating bike, cropped from the full-resolution upload. Common misreads (O/0, I/1, S/5, B/8…) are corrected according to whether the plate format expects a letter or a digit at that position, and the result is accepted only if it has a valid Indian state code. Otherwise the plate is `Not detected`.
- **Video duplicate merging.** Every Nth frame is analysed (default 10; set it with `?frame_interval=`). A sighting joins an existing unique violation when:
  - it is the **same violation type**,
  - its motorcycle box **overlaps** (IoU ≥ 0.3) or its centre is **close** to the previous sighting's, and
  - it is **within 2 seconds** of that sighting.

  After OCR, violations of the same type with the **same readable plate** are also merged. Each unique violation keeps its **best evidence frame**, ranked by violation confidence, detection confidence and whether the plate was readable.

## Setup
Requires Python 3.11+. Run from the `trafficeye/` folder.

```bash
python -m venv .venv
# Windows ("py -m venv .venv" if python is not on PATH)
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.seed        # optional demo data
uvicorn backend.main:app --reload
```

- UI: http://localhost:8000 (Dashboard at `/dashboard`, History at `/history`)
- API docs: http://localhost:8000/docs

The first start downloads `yolov8n.pt` (about 6 MB) and the EasyOCR weights (about 100 MB) into `backend/models/`.

## Helmet model (optional)
Helmet detection needs a YOLOv8 **detection** model with helmet / no-helmet classes at `backend/models/helmet.pt`. Without it the app runs normally and only skips the helmet check.

Search [Hugging Face](https://huggingface.co/models?search=helmet) or [Roboflow Universe](https://universe.roboflow.com/search?q=helmet) for helmet / no-helmet models trained on motorcycle riders, and download the `.pt` weights. The team tested [`iam-tsr/yolov8n-helmet-detection`](https://huggingface.co/iam-tsr/yolov8n-helmet-detection) (MIT licence), a community model and not an official Ultralytics one:

```bash
# Linux/macOS  (Windows PowerShell: use curl.exe and backend\models\helmet.pt)
curl -L -o backend/models/helmet.pt https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt
```

Always check the class names, because model cards are not always accurate:

```python
from ultralytics import YOLO
print(YOLO("backend/models/helmet.pt").names)   # {0: 'With Helmet', 1: 'Without Helmet'}
```

Restart the server, then open `/api/health`. `helmet_class_mapping` must show every class as `HELMET` or `NO_HELMET`. If one shows `IGNORED`, add its lower-case name, without spaces, `-` or `_`, to `HELMET_CLASS_KEYS` or `NO_HELMET_CLASS_KEYS` in `backend/config.py`.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model status and helmet class mapping |
| POST | `/api/detect/image` | Analyse a JPG/PNG |
| POST | `/api/detect/video?frame_interval=10` | Analyse an MP4 (duplicates merged) |
| GET | `/api/violations?type=&date=&search=` | History; filters can be combined |
| GET | `/api/report/{id}` | Download the PDF e-challan (`?download=false` opens it in the browser) |
| DELETE | `/api/violations/{id}` | Delete a record, its challan, and its evidence image if no other record uses it |
| GET | `/api/stats` | Dashboard totals and 7-day trend |

### Sample responses
`POST /api/detect/video` (trimmed):
```json
{
  "success": true,
  "frames_processed": 12,
  "candidates_found": 22,
  "violations": [
    {
      "id": 45, "type": "TRIPLE_RIDING", "confidence": 0.82,
      "plate_number": "Not detected", "fine": 1000,
      "time_seconds": 4.0, "merged_detections": 10,
      "evidence_url": "/evidence/video_20260925_125501_1a2b3c4d5e.jpg",
      "report_url": "/api/report/45"
    }
  ]
}
```

`GET /api/stats` (trimmed):
```json
{ "success": true, "total": 20, "no_helmet": 13, "triple_riding": 7, "total_fines": 20000,
  "daily_trend": [{ "date": "2026-09-25", "count": 4, "no_helmet": 3, "triple_riding": 1 }] }
```

Every error:
```json
{ "success": false, "error": "Unsupported file type", "detail": "Only MP4 videos are allowed." }
```

## Limitations
- `yolov8n` is small and fast. On crowded bikes it can merge overlapping riders into one box, so some triple riding is missed.
- Helmet accuracy depends entirely on the helmet model. Small, distant heads are often `UNKNOWN`, which is reported as no violation.
- Plate OCR needs a reasonably sharp, front- or rear-facing plate. Side views return `Not detected`.
- Video merging is lightweight box matching, not a full multi-object tracker. It runs on CPU, so long or high-resolution videos take time.
- Challans are demo documents and not legally valid.

## Future Scope
Planned features, grouped by category.

**AI Detection**
- Mobile phone use while driving
- Seat belt violation
- Wrong-side driving
- Red-light jumping
- Stop-line violation
- Overspeed detection
- Fine-tuned number plate detector ahead of OCR

**Monitoring**
- Live CCTV / camera feed
- GPU deployment with multi-camera dashboards

**Review**
- Officer approve / reject workflow with an audit log

**Analytics**
- Violation hotspot map
- Peak violation time
- Repeat offender detection

**Alerts**
- Email / SMS challan notification

**Security**
- Admin login and role-based access

## Team members
- Gajavanan.R
- Kiran Raaj G.L
