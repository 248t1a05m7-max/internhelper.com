# Face Recognition Attendance System for Dhanekula College

A Flask-based attendance platform with role-based access:

- **Admin module**: secure login, student management, multi-sample face dataset capture, training trigger, attendance reports, filtering, CSV/PDF export.
- **Student module**: secure login, face-based **Time-In / Time-Out**, duplicate prevention window, personal daily/weekly/monthly logs.

## Tech Stack
- Python + Flask
- OpenCV + face_recognition + NumPy
- SQLite
- HTML/CSS/JS templates
- reportlab for PDF export

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open:
- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/login/admin`
- `http://127.0.0.1:5000/login/student`

## Default Admin Credentials
- Username: `admin`
- Password: `admin123`

You can override using environment variables:
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `APP_SECRET_KEY`

## Key Functional Coverage
- Secure admin and student authentication
- Register/update/delete students (ID, name, department)
- Capture **multiple face samples** (minimum 3) from webcam
- Train trigger endpoint (latest samples are active)
- Real-time face recognition attendance with timestamp
- Time-In / Time-Out actions
- Duplicate prevention (same action blocked within 10 minutes)
- Reports with date/student filters
- Export reports to CSV/PDF

## Notes
- Timestamps are stored in UTC.
- Ensure camera permission is allowed in browser.
- face_recognition requires native build prerequisites in some systems.
