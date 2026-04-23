import base64
import csv
import io
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "attendance.db"
MATCH_TOLERANCE = 0.45
DUPLICATE_WINDOW_MINUTES = 10

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("APP_SECRET_KEY", secrets.token_hex(24))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin')),
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS face_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            face_encoding BLOB NOT NULL,
            captured_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('in', 'out')),
            status TEXT NOT NULL,
            note TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    row = cur.execute("SELECT id FROM users WHERE username = ?", (admin_username,)).fetchone()
    if not row:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
            (admin_username, generate_password_hash(admin_password), utc_now()),
        )

    conn.commit()
    conn.close()


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def decode_image(data_url: str):
    if not data_url or "," not in data_url:
        return None
    image_data = base64.b64decode(data_url.split(",", 1)[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(image)


def encode_to_blob(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float32).tobytes()


def blob_to_encoding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def parse_face_samples(raw_payload: str):
    if not raw_payload:
        return []
    return [item for item in raw_payload.split("||") if item.strip()]


def extract_single_encoding(face_data: str):
    rgb = decode_image(face_data)
    if rgb is None:
        return None, "Invalid face image payload"

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    if len(encodings) != 1:
        return None, "Each sample must contain exactly one face"

    return encodings[0], None


def admin_required():
    return session.get("role") == "admin"


def student_required():
    return session.get("role") == "student" and session.get("student_id")


def has_recent_duplicate(student_id: str, action: str):
    since = (datetime.utcnow() - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat(timespec="seconds")
    conn = get_db()
    row = conn.execute(
        """
        SELECT id FROM attendance
        WHERE student_id = ? AND action = ? AND status = 'success' AND timestamp >= ?
        ORDER BY timestamp DESC LIMIT 1
        """,
        (student_id, action, since),
    ).fetchone()
    conn.close()
    return row is not None


def mark_attendance(student_id: str, mode: str, action: str, status: str, note: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO attendance (student_id, mode, action, status, note, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (student_id, mode, action, status, note, utc_now()),
    )
    conn.commit()
    conn.close()


def fetch_student_encodings(student_id: str):
    conn = get_db()
    rows = conn.execute("SELECT face_encoding FROM face_samples WHERE student_id = ?", (student_id,)).fetchall()
    conn.close()
    return [blob_to_encoding(row["face_encoding"]) for row in rows]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"ok": True, "service": "face-attendance-system"}


@app.route("/login/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND role = 'admin'", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["role"] = "admin"
            session["username"] = user["username"]
            return redirect(url_for("admin_dashboard"))

        return render_template("login_admin.html", error="Invalid admin credentials")

    return render_template("login_admin.html")


@app.route("/login/student", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        student = conn.execute("SELECT id, name, password_hash FROM students WHERE id = ?", (student_id,)).fetchone()
        conn.close()

        if student and check_password_hash(student["password_hash"], password):
            session.clear()
            session["role"] = "student"
            session["student_id"] = student["id"]
            session["student_name"] = student["name"]
            return redirect(url_for("student_dashboard"))

        return render_template("login_student.html", error="Invalid student ID or password")

    return render_template("login_student.html")


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.get("/admin")
def admin_dashboard():
    if not admin_required():
        return redirect(url_for("admin_login"))

    date_filter = request.args.get("date", "").strip()
    student_filter = request.args.get("student_id", "").strip()

    conn = get_db()
    students = conn.execute(
        """
        SELECT s.id, s.name, s.department, s.created_at,
               (SELECT COUNT(1) FROM face_samples fs WHERE fs.student_id = s.id) AS sample_count
        FROM students s ORDER BY s.created_at DESC
        """
    ).fetchall()

    query = (
        "SELECT a.student_id, s.name, s.department, a.mode, a.action, a.status, a.note, a.timestamp "
        "FROM attendance a JOIN students s ON s.id = a.student_id WHERE 1=1"
    )
    params = []
    if date_filter:
        query += " AND substr(a.timestamp, 1, 10) = ?"
        params.append(date_filter)
    if student_filter:
        query += " AND a.student_id = ?"
        params.append(student_filter)
    query += " ORDER BY a.timestamp DESC LIMIT 500"

    attendance_rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()

    return render_template(
        "admin.html",
        students=students,
        attendance=attendance_rows,
        date_filter=date_filter,
        student_filter=student_filter,
    )


@app.post("/admin/student/save")
def admin_save_student():
    if not admin_required():
        return redirect(url_for("admin_login"))

    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    password = request.form.get("password", "").strip() or "student123"
    face_samples_payload = request.form.get("face_samples", "")

    if not student_id or not name or not department:
        return redirect(url_for("admin_dashboard", error="ID, name, and department are required"))

    face_samples = parse_face_samples(face_samples_payload)
    if len(face_samples) < 3:
        return redirect(url_for("admin_dashboard", error="Capture at least 3 face samples"))

    extracted = []
    for sample in face_samples:
        encoding, err = extract_single_encoding(sample)
        if err:
            return redirect(url_for("admin_dashboard", error=err))
        extracted.append(encoding)

    now = utc_now()
    conn = get_db()
    existing = conn.execute("SELECT id FROM students WHERE id = ?", (student_id,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE students SET name = ?, department = ?, updated_at = ? WHERE id = ?",
            (name, department, now, student_id),
        )
    else:
        conn.execute(
            "INSERT INTO students (id, name, department, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (student_id, name, department, generate_password_hash(password), now, now),
        )

    conn.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
    for enc in extracted:
        conn.execute(
            "INSERT INTO face_samples (student_id, face_encoding, captured_at) VALUES (?, ?, ?)",
            (student_id, encode_to_blob(enc), now),
        )

    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard", message=f"Student {name} saved with {len(extracted)} samples"))


@app.post("/admin/student/delete/<student_id>")
def admin_delete_student(student_id: str):
    if not admin_required():
        return redirect(url_for("admin_login"))

    conn = get_db()
    conn.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard", message=f"Deleted student {student_id}"))


@app.post("/admin/train")
def admin_train():
    if not admin_required():
        return redirect(url_for("admin_login"))
    return redirect(url_for("admin_dashboard", message="Training complete. Latest samples are active."))


def attendance_query(date_filter: str, student_filter: str):
    query = (
        "SELECT a.student_id, s.name, s.department, a.mode, a.action, a.status, a.note, a.timestamp "
        "FROM attendance a JOIN students s ON s.id = a.student_id WHERE 1=1"
    )
    params = []
    if date_filter:
        query += " AND substr(a.timestamp, 1, 10) = ?"
        params.append(date_filter)
    if student_filter:
        query += " AND a.student_id = ?"
        params.append(student_filter)
    query += " ORDER BY a.timestamp DESC"
    return query, tuple(params)


@app.get("/admin/export/csv")
def export_csv():
    if not admin_required():
        return redirect(url_for("admin_login"))

    date_filter = request.args.get("date", "").strip()
    student_filter = request.args.get("student_id", "").strip()
    query, params = attendance_query(date_filter, student_filter)

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Department", "Mode", "Action", "Status", "Note", "Timestamp (UTC)"])
    for row in rows:
        writer.writerow(
            [
                row["student_id"],
                row["name"],
                row["department"],
                row["mode"],
                row["action"],
                row["status"],
                row["note"],
                row["timestamp"],
            ]
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{datetime.utcnow().date().isoformat()}.csv"},
    )


@app.get("/admin/export/pdf")
def export_pdf():
    if not admin_required():
        return redirect(url_for("admin_login"))

    date_filter = request.args.get("date", "").strip()
    student_filter = request.args.get("student_id", "").strip()
    query, params = attendance_query(date_filter, student_filter)

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(30, y, "Dhanekula College - Attendance Report")
    y -= 20
    pdf.setFont("Helvetica", 9)
    pdf.drawString(30, y, f"Generated (UTC): {utc_now()}")
    y -= 20

    for idx, row in enumerate(rows, start=1):
        line = f"{idx}. {row['student_id']} | {row['name']} | {row['action']} | {row['status']} | {row['timestamp']}"
        pdf.drawString(30, y, line[:110])
        y -= 14
        if y < 40:
            pdf.showPage()
            y = height - 40

    pdf.save()
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=attendance_{datetime.utcnow().date().isoformat()}.pdf"},
    )


@app.get("/student")
def student_dashboard():
    if not student_required():
        return redirect(url_for("student_login"))

    scope = request.args.get("scope", "daily")
    allowed = {"daily": 1, "weekly": 7, "monthly": 30}
    days = allowed.get(scope, 1)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")

    conn = get_db()
    logs = conn.execute(
        """
        SELECT mode, action, status, note, timestamp
        FROM attendance
        WHERE student_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        """,
        (session["student_id"], since),
    ).fetchall()
    conn.close()

    return render_template("student.html", logs=logs, scope=scope)


@app.post("/api/mark-face")
def mark_face():
    if not student_required():
        return jsonify({"ok": False, "message": "Unauthorized"}), 401

    action = request.form.get("action", "in").strip().lower()
    face_data = request.form.get("face_data", "")

    if action not in {"in", "out"}:
        return jsonify({"ok": False, "message": "Invalid action"}), 400

    student_id = session["student_id"]
    known_encodings = fetch_student_encodings(student_id)
    if not known_encodings:
        return jsonify({"ok": False, "message": "No training data found. Contact admin."}), 400

    rgb = decode_image(face_data)
    if rgb is None:
        return jsonify({"ok": False, "message": "Face image missing"}), 400

    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    locations = face_recognition.face_locations(small, model="hog")
    encodings = face_recognition.face_encodings(small, locations)

    if len(encodings) != 1:
        mark_attendance(student_id, "face", action, "failed", "No clear single face")
        return jsonify({"ok": False, "message": "Need exactly one clear face in frame."}), 400

    distances = face_recognition.face_distance(known_encodings, encodings[0])
    min_distance = float(np.min(distances)) if len(distances) else 1.0

    if min_distance > MATCH_TOLERANCE:
        mark_attendance(student_id, "face", action, "failed", f"distance={min_distance:.3f}")
        return jsonify({"ok": False, "message": "Face recognition failed."}), 401

    if has_recent_duplicate(student_id, action):
        return jsonify(
            {
                "ok": False,
                "message": f"Duplicate prevented: already marked {action.upper()} in last {DUPLICATE_WINDOW_MINUTES} mins.",
            }
        )

    mark_attendance(student_id, "face", action, "success", f"distance={min_distance:.3f}")
    return jsonify({"ok": True, "message": f"{action.upper()} marked successfully at {utc_now()} UTC"})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
