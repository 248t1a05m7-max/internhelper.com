# Smart Attendance Website (Face + QR Fallback)

This project is now a **website** using Flask.

- **Admin page**: register students with webcam face capture, see students, and view attendance logs.
- **Student page**: mark attendance with face recognition; if face fails, use **QR fallback**.

## 1) Run as a website on your laptop

### Prerequisites
- Python 3.10+
- Webcam
- Build tools required by `face_recognition`/`dlib`:
  - **Ubuntu/Debian**: `cmake`, `build-essential`, `python3-dev`
  - **Windows**: Visual Studio C++ Build Tools + CMake

### Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Start website
```bash
python app.py
```

Open:
- `http://127.0.0.1:5000/` (home)
- `http://127.0.0.1:5000/admin` (admin panel)
- `http://127.0.0.1:5000/student` (student panel)

Health check:
- `http://127.0.0.1:5000/health`

---

## 2) “Compile” / package for production (Docker way)

If by “compile” you mean package for deployment, use Docker.

### Build image
```bash
docker build -t smart-attendance:latest .
```

### Run container
```bash
docker run --rm -p 5000:5000 -v $(pwd)/data:/app/data smart-attendance:latest
```

Then open `http://127.0.0.1:5000`.

---

## 3) Deploy as real website (public)

Quick options:
1. **Render / Railway / Fly.io**
2. **AWS EC2 / DigitalOcean VPS** with Nginx + Gunicorn

### Gunicorn example (Linux server)
```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:8000 app:app
```

Use Nginx reverse proxy from port 80/443 to `localhost:8000`.

---

## 4) How students/admin use it

1. Admin opens `/admin`.
2. Admin enters ID/Name/Email and captures face from webcam.
3. Student opens `/student`, selects ID, and marks attendance via face.
4. If face fails, student uses QR link generated for their ID.
5. Admin checks attendance table in `/admin`.

---

## 5) Common issues

### `ModuleNotFoundError: cv2`
Install OpenCV:
```bash
pip install opencv-python
```

### `ModuleNotFoundError: face_recognition`
Install build deps first (`cmake`, compiler, python-dev), then:
```bash
pip install face-recognition
```

### Camera not opening in browser
- Use HTTPS in production (browser camera policy).
- Allow camera permissions in browser settings.

---

## 6) Project structure

- `app.py` — Flask website backend (admin/student/QR routes)
- `templates/` — HTML pages
- `static/` — CSS + JS (webcam capture)
- `data/` — SQLite DB and generated QR files
- `attendance_system.py` — standalone CLI version
