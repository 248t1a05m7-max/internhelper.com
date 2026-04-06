import csv
import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pyttsx3

DATA_DIR = Path("data")
ENCODINGS_FILE = DATA_DIR / "encodings.npz"
STUDENTS_FILE = DATA_DIR / "students.csv"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"

# Recognition tuning (fast + stable)
FRAME_SCALE = 0.5  # 50% frame for faster detection
PROCESS_EVERY_N_FRAMES = 2
MATCH_TOLERANCE = 0.45  # lower means stricter recognition

# Voice engine
engine = pyttsx3.init()
engine.setProperty("rate", 175)


def speak(text: str) -> None:
    """Speak short feedback without blocking too long."""
    engine.say(text)
    engine.runAndWait()


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not STUDENTS_FILE.exists():
        with STUDENTS_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "email"])

    if not ATTENDANCE_FILE.exists():
        with ATTENDANCE_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "date", "time"])


def send_email(to_email: str, name: str, date: str, time_now: str) -> None:
    """Send attendance confirmation email using env vars."""
    sender_email = os.getenv("ATTENDANCE_EMAIL")
    app_password = os.getenv("ATTENDANCE_APP_PASSWORD")

    if not sender_email or not app_password or not to_email:
        return

    subject = "Attendance Confirmation"
    body = (
        f"Hello {name},\n\n"
        "Your attendance has been marked successfully.\n\n"
        f"Date: {date}\n"
        f"Time: {time_now}\n\n"
        "Regards,\n"
        "College Attendance System"
    )

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        print("Email Error:", e)


def load_students() -> dict:
    students = {}
    if not STUDENTS_FILE.exists():
        return students

    with STUDENTS_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            students[row["id"]] = {"name": row["name"], "email": row["email"]}
    return students


def save_student(student_id: str, name: str, email: str) -> None:
    students = load_students()
    students[student_id] = {"name": name, "email": email}

    with STUDENTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "email"])
        for sid, info in students.items():
            writer.writerow([sid, info["name"], info["email"]])


def load_encodings():
    if ENCODINGS_FILE.exists():
        data = np.load(ENCODINGS_FILE, allow_pickle=True)
        return list(data["ids"]), list(data["encodings"])
    return [], []


def save_encodings(ids, encodings) -> None:
    np.savez_compressed(
        ENCODINGS_FILE,
        ids=np.array(ids, dtype=object),
        encodings=np.array(encodings, dtype=np.float32),
    )


def draw_ui(frame, title: str, subtitle: str, color=(36, 255, 12)):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
    cv2.putText(frame, subtitle, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)


def register_student() -> None:
    ensure_data_files()
    student_id = input("Enter student ID: ").strip()
    name = input("Enter student name: ").strip()
    email = input("Enter student email (optional): ").strip()

    if not student_id or not name:
        print("Student ID and name are required.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera.")
        return

    print("Registration started. Look at camera and press C to capture face.")
    speak("Registration started. Please look at the camera.")

    captured_encoding = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        draw_ui(
            frame,
            "Student Registration (No image upload)",
            "Press C to capture | Q to cancel",
            color=(0, 200, 255),
        )

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 255), 2)

        cv2.imshow("Register Student", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            if len(locations) != 1:
                print("Please ensure exactly one face is visible.")
                speak("Please keep one face in frame.")
                continue

            encoding = face_recognition.face_encodings(rgb, locations)
            if not encoding:
                print("Could not extract face encoding. Try again.")
                continue

            captured_encoding = encoding[0]  # 128D embedding
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_encoding is None:
        print("Registration cancelled.")
        return

    ids, encodings = load_encodings()
    if student_id in ids:
        idx = ids.index(student_id)
        encodings[idx] = captured_encoding
    else:
        ids.append(student_id)
        encodings.append(captured_encoding)

    save_encodings(ids, encodings)
    save_student(student_id, name, email)

    print(f"Student {name} registered successfully using 128D face encoding.")
    speak(f"Registration complete for {name}")


def already_marked_today(student_id: str, date: str) -> bool:
    if not ATTENDANCE_FILE.exists():
        return False

    with ATTENDANCE_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == student_id and row["date"] == date:
                return True
    return False


def mark_attendance(student_id: str, name: str) -> None:
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    if already_marked_today(student_id, date):
        print("Attendance already marked today.")
        speak("Attendance already marked for today.")
        return

    with ATTENDANCE_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, name, date, time_now])

    print(f"Attendance marked for {name} at {time_now}.")
    speak(f"Welcome {name}. Attendance marked successfully.")

    students = load_students()
    email = students.get(student_id, {}).get("email", "")
    send_email(email, name, date, time_now)


def run_recognition() -> None:
    ensure_data_files()
    students = load_students()
    known_ids, known_encodings = load_encodings()

    if not known_encodings:
        print("No registered students found. Please register first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera.")
        return

    print("Recognition started. Press Q to exit.")
    process_counter = 0
    last_spoken = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        draw_ui(
            frame,
            "Fast Face Attendance (128D)",
            "Look at camera | Press Q to quit",
            color=(36, 255, 12),
        )

        if process_counter % PROCESS_EVERY_N_FRAMES == 0:
            locations = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, locations)

            for encoding, (top, right, bottom, left) in zip(encodings, locations):
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx = int(np.argmin(distances))
                best_distance = float(distances[best_idx])

                # Scale back to original frame
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)

                if best_distance <= MATCH_TOLERANCE:
                    student_id = known_ids[best_idx]
                    student = students.get(student_id, {"name": "Unknown"})
                    name = student["name"]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{name} ({best_distance:.2f})",
                        (left, top - 12),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    mark_attendance(student_id, name)
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Face not recognized",
                        (left, top - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    if time.time() - last_spoken > 3:
                        speak("Face not clear. Please look at the camera.")
                        last_spoken = time.time()

        process_counter += 1
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ensure_data_files()
    print("\n=== Smart Attendance System ===")
    print("1. Register Student (camera only)")
    print("2. Mark Attendance (face recognition)")
    choice = input("Select option (1/2): ").strip()

    if choice == "1":
        register_student()
    elif choice == "2":
        run_recognition()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
