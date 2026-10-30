# AI Smart Classroom Attendance System

A production-structured facial-recognition attendance system built with
Streamlit, OpenCV, InsightFace (RetinaFace detection + ArcFace recognition,
**pretrained only, no training**), SQLAlchemy/SQLite, and pandas.

## 1. Features

- Student registration with live iPhone/webcam capture, automatic face
  detection, and image-quality gatekeeping (blur, brightness, face count,
  face size).
- One-click classroom attendance: detect every face, recognize each
  against a cached embedding gallery via cosine similarity, auto-mark
  Present / Absent / Unknown, and persist a full session.
- Dashboard with live metrics and charts (daily trend, department
  breakdown, per-student attendance).
- Search & filter across students and attendance history.
- Daily / Weekly / Monthly / Semester reports, exportable as CSV, Excel
  (styled), or PDF.
- Admin tools: delete/update/deactivate students, recapture images,
  delete attendance sessions, database backup & restore.
- Configurable recognition threshold, detector confidence, and camera
  resolution via the Settings page (persisted to `database/settings.json`).
- Centralized rotating-file logging (`logs/app.log`) for every error.

## 2. Project Structure

```
attendance_project/
├── app.py                     # Streamlit entry point + sidebar router
├── config.py                  # All paths, thresholds, camera defaults
├── database.py                # SQLAlchemy engine/session/backup/restore
├── requirements.txt
├── README.md
├── dataset/<student_id>/      # Cropped face images per student
├── embeddings/                # (reserved for optional .npy backups)
├── reports/                   # Exported CSV/Excel/PDF + classroom photos
├── database/                  # attendance.db, settings.json, backups/
├── logs/                      # app.log (rotating)
├── models/
│   └── models.py              # Teacher, Course, Class, Student, Embedding,
│                               # Attendance, AttendanceDetail ORM models
├── recognition/
│   ├── face_detector.py       # RetinaFace wrapper (via InsightFace)
│   └── face_recognizer.py     # ArcFace embeddings + cosine-similarity matching
├── services/
│   ├── embedding_store.py     # Cached in-memory embedding gallery
│   ├── registration_service.py
│   ├── attendance_service.py
│   ├── report_service.py
│   └── settings_service.py
├── utils/
│   ├── logger.py
│   └── image_quality.py       # Blur / brightness / face-count / size checks
└── pages/
    ├── home.py, dashboard.py, student_registration.py, student_list.py,
    └── take_attendance.py, attendance_history.py, reports.py, settings_page.py
```

## 3. Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on InsightFace / RetinaFace models:** the first time `FaceDetector`
> or `FaceRecognizer` is used, InsightFace automatically downloads the
> `buffalo_l` model pack (RetinaFace detector + ArcFace recognizer, both
> pretrained) into `models_weights/`. This requires internet access on
> first run only; after that it's fully offline.

## 4. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). On a
phone, use "Add to Home Screen" from Safari/Chrome pointed at your
machine's LAN IP (e.g. `http://192.168.x.x:8501`) so the iPhone's own
camera is used by `st.camera_input`.

## 5. Typical Workflow

1. **Student Registration** — fill in the student's details, then capture
   20–50 images (a new photo per capture). Frames are auto-rejected with a
   clear reason if blurry, too dark, no face, or multiple faces; accepted
   frames are cropped and their embedding is cached immediately.
2. **Take Attendance** — a teacher optionally selects a class, captures one
   classroom photo, and the system detects, recognizes, and marks
   Present/Absent/Unknown automatically, saving a full session.
3. **Attendance History / Reports** — search, filter, and export the
   results in CSV, Excel, or PDF.
4. **Settings** — tune the recognition threshold if you're seeing too many
   false matches (raise it) or too many missed matches (lower it), and
   back up/restore the database.

## 6. Design Notes

- **No CNN training.** Recognition uses InsightFace's pretrained ArcFace
  embeddings exclusively; only cosine-similarity comparison happens at
  runtime.
- **Embeddings are cached, never recomputed for known faces** — they're
  generated once at registration and stored as raw bytes in the
  `embeddings` table; `EmbeddingGallery` loads them into memory once and
  is explicitly invalidated after any registration/deletion.
- **Duplicate detections** in a classroom photo are suppressed via IoU
  overlap filtering in `FaceDetector`.
- **Every module uses type hints, docstrings, and a shared logger** — see
  `utils/logger.py`; all exceptions are logged with stack traces to
  `logs/app.log`.

## 7. Extending (Future Features from the Spec)

The current architecture is intentionally layered (`recognition/` →
`services/` → `pages/`) so the following can be added without touching the
UI layer: real-time video attendance (swap `st.camera_input` for a video
stream loop calling `AttendanceService` per frame), multiple classrooms
(the `Class`/`Course`/`Teacher` tables already support this), teacher/
student login (add an `auth` module + guard in `app.py`), liveness /
anti-spoofing (add a check inside `FaceDetector.detect` before accepting a
box), and cloud database (swap `DATABASE_URL` in `config.py` for a Postgres
DSN — SQLAlchemy handles the rest).
