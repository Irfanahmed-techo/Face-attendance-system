import cv2
import face_recognition
import pickle
from datetime import datetime
import os
import time
import sqlite3
import numpy as np
from spoof_check import movement_detected

print("Attendance file started")

# =========================
# Load registered encodings
# =========================
with open("data/encodings.pkl", "rb") as f:
    users = pickle.load(f)

known_encodings = []
known_names = []

for user in users:
    for enc in user["encodings"]:
        known_encodings.append(enc)
        known_names.append(user["name"])

# =========================
# Database setup
# =========================
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/attendance.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT,
    status TEXT
)
""")

conn.commit()
conn.close()

# =========================
# Camera setup
# =========================
cap = cv2.VideoCapture(0)
prev_frame = None
marked_today = set()
face_seen_time = {}

REQUIRED_TIME = 2          # seconds face must stay visible
THRESHOLD = 0.45           # strict threshold for recognition

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)

    # -------------------------
    # Spoof check (only if face detected)
    # -------------------------
    if prev_frame is not None and len(face_locations) > 0:
        if not movement_detected(prev_frame, frame):
            cv2.putText(
                frame,
                "Spoof detected! Please move slightly",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.imshow("Attendance", frame)
            prev_frame = frame.copy()
            if cv2.waitKey(1) == 27:
                break
            continue

    # -------------------------
    # Face recognition
    # -------------------------
    if len(face_locations) > 0:
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for enc in face_encs:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = np.argmin(distances)

            # ========= REGISTERED FACE =========
            if distances[best_match_index] < THRESHOLD:
                name = known_names[best_match_index]

                if name in marked_today:
                    continue

                now = time.time()

                if name not in face_seen_time:
                    face_seen_time[name] = now
                    cv2.putText(
                        frame,
                        "Hold still...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 0),
                        2
                    )
                elif now - face_seen_time[name] >= REQUIRED_TIME:
                    date = datetime.now().strftime("%Y-%m-%d")
                    time_now = datetime.now().strftime("%H:%M:%S")

                    conn = sqlite3.connect("data/attendance.db")
                    cur = conn.cursor()

                    cur.execute(
                        "SELECT * FROM attendance WHERE name=? AND date=?",
                        (name, date)
                    )
                    today = cur.fetchall()

                    status = "Punch-In" if len(today) == 0 else "Punch-Out"

                    cur.execute(
                        "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                        (name, date, time_now, status)
                    )

                    conn.commit()
                    conn.close()

                    print(name, status)
                    marked_today.add(name)

                    cv2.putText(
                        frame,
                        f"{name} {status}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            # ========= UNREGISTERED FACE =========
            else:
                cv2.putText(
                    frame,
                    "Face not recognized",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

    cv2.imshow("Attendance", frame)
    prev_frame = frame.copy()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
