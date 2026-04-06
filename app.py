import base64
import io
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image

try:
    import qrcode
except Exception:  # pragma: no cover
    qrcode = None

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "attendance.db"
QR_DIR = BASE_DIR / "data" / "qrcodes"
MATCH_TOLERANCE = 0.45

app = Flask(__name__)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    QR_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            face_encoding BLOB NOT NULL,
            qr_token TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """
    )
    conn.commit()
    conn.close()


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


def mark_attendance(student_id: str, mode: str, status: str, note: str = ""):
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn = get_db()
    conn.execute(
        "INSERT INTO attendance (student_id, mode, status, note, timestamp) VALUES (?, ?, ?, ?, ?)",
        (student_id, mode, status, note, now),
    )
    conn.commit()
    conn.close()


def has_marked_today(student_id: str) -> bool:
    today = datetime.utcnow().date().isoformat()
    conn = get_db()
    row = conn.execute(
        "SELECT 1 FROM attendance WHERE student_id = ? AND substr(timestamp, 1, 10) = ? AND status = 'success'",
        (student_id, today),
    ).fetchone()
    conn.close()
    return row is not None


def create_qr(student_id: str, token: str):
    qr_payload = url_for("qr_checkin", token=token, _external=True)
    output = QR_DIR / f"{student_id}.png"

    if qrcode is None:
        return qr_payload, None

    img = qrcode.make(qr_payload)
    img.save(output)
    return qr_payload, output


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"ok": True, "service": "smart-attendance"}


@app.get("/admin")
def admin():
    conn = get_db()
    students = conn.execute("SELECT id, name, email, created_at FROM students ORDER BY created_at DESC").fetchall()
    attendance = conn.execute(
        """
        SELECT a.student_id, s.name, a.mode, a.status, a.note, a.timestamp
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.timestamp DESC
        LIMIT 100
        """
    ).fetchall()
    conn.close()
    return render_template("admin.html", students=students, attendance=attendance)


@app.post("/admin/register")
def admin_register():
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    face_data = request.form.get("face_data", "")

    if not student_id or not name:
        return redirect(url_for("admin", error="Student ID and name are required"))

    rgb = decode_image(face_data)
    if rgb is None:
        return redirect(url_for("admin", error="Face capture missing"))

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    if len(encodings) != 1:
        return redirect(url_for("admin", error="Please capture exactly one clear face"))

    token = secrets.token_urlsafe(18)
    encoding_blob = encode_to_blob(encodings[0])

    conn = get_db()
    conn.execute(
        """
        INSERT INTO students (id, name, email, face_encoding, qr_token, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            email=excluded.email,
            face_encoding=excluded.face_encoding,
            qr_token=excluded.qr_token
        """,
        (student_id, name, email, encoding_blob, token, datetime.utcnow().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()

    create_qr(student_id, token)
    return redirect(url_for("admin", message=f"Student {name} registered successfully"))


@app.get("/admin/qr/<student_id>")
def get_qr(student_id: str):
    img = QR_DIR / f"{student_id}.png"
    if img.exists():
        return send_file(img, mimetype="image/png")

    conn = get_db()
    row = conn.execute("SELECT qr_token FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    if not row:
        return "Student not found", 404
    payload, _ = create_qr(student_id, row["qr_token"])
    if img.exists():
        return send_file(img, mimetype="image/png")
    return jsonify({"qr_url": payload})


@app.get("/student")
def student_page():
    conn = get_db()
    students = conn.execute("SELECT id, name FROM students ORDER BY name").fetchall()
    conn.close()
    return render_template("student.html", students=students)


@app.post("/api/mark-face")
def mark_face():
    student_id = request.form.get("student_id", "").strip()
    face_data = request.form.get("face_data", "")

    conn = get_db()
    student = conn.execute("SELECT id, name, face_encoding FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()

    if not student:
        return jsonify({"ok": False, "message": "Student not found"}), 404

    rgb = decode_image(face_data)
    if rgb is None:
        return jsonify({"ok": False, "message": "Face image missing"}), 400

    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    locations = face_recognition.face_locations(small, model="hog")
    encodings = face_recognition.face_encodings(small, locations)

    if len(encodings) != 1:
        mark_attendance(student_id, "face", "failed", "No clear single face")
        return jsonify({"ok": False, "message": "Face not clear. Try again or use QR fallback."}), 400

    known = blob_to_encoding(student["face_encoding"])
    dist = float(face_recognition.face_distance([known], encodings[0])[0])

    if dist <= MATCH_TOLERANCE:
        if has_marked_today(student_id):
            return jsonify({"ok": True, "message": "Attendance already marked today."})
        mark_attendance(student_id, "face", "success", f"distance={dist:.3f}")
        return jsonify({"ok": True, "message": f"Welcome {student['name']}! Face attendance marked."})

    mark_attendance(student_id, "face", "failed", f"distance={dist:.3f}")
    return jsonify({"ok": False, "message": "Face recognition failed. Use QR fallback."}), 401


@app.get("/student/qr-checkin")
def qr_checkin():
    token = request.args.get("token", "")
    conn = get_db()
    student = conn.execute("SELECT id, name FROM students WHERE qr_token = ?", (token,)).fetchone()
    conn.close()

    if not student:
        return render_template("qr_result.html", ok=False, message="Invalid QR token")

    if has_marked_today(student["id"]):
        return render_template("qr_result.html", ok=True, message="Attendance already marked today.")

    mark_attendance(student["id"], "qr", "success", "QR fallback")
    return render_template("qr_result.html", ok=True, message=f"Welcome {student['name']}! QR attendance marked.")


init_db()


if __name__ == "__main__":
    app.run(debug=True)
