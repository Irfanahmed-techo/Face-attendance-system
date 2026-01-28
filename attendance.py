import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os
import time
from spoof_check import movement_detected

print("Attendance file started")

# Load registered encodings
with open("data/encodings.pkl", "rb") as f:
    users = pickle.load(f)

known_encodings = []
known_names = []

for user in users:
    for enc in user["encodings"]:
        known_encodings.append(enc)
        known_names.append(user["name"])

# Prepare attendance file
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(
        "data/attendance.csv", index=False
    )

cap = cv2.VideoCapture(0)
prev_frame = None
marked_today = set()

# ⏳ delay control
face_seen_time = {}
REQUIRED_TIME = 2  # seconds face must stay visible

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces first
    face_locations = face_recognition.face_locations(rgb)

    # Spoof check only when face exists
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

    if len(face_locations) > 0:
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for enc in face_encs:
            matches = face_recognition.compare_faces(known_encodings, enc)

            if True in matches:
                name = known_names[matches.index(True)]

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
                else:
                    if now - face_seen_time[name] >= REQUIRED_TIME:
                        date = datetime.now().strftime("%Y-%m-%d")
                        time_now = datetime.now().strftime("%H:%M:%S")

                        df = pd.read_csv("data/attendance.csv")
                        today = df[(df["Name"] == name) & (df["Date"] == date)]

                        status = "Punch-In" if len(today) == 0 else "Punch-Out"

                        df.loc[len(df)] = [name, date, time_now, status]
                        df.to_csv("data/attendance.csv", index=False)

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

    cv2.imshow("Attendance", frame)
    prev_frame = frame.copy()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
