import cv2
import face_recognition
import pickle
import os

print("Register file started")

name = input("Enter your name: ")

cap = cv2.VideoCapture(0)
encodings = []

print("Press SPACE to capture face, ESC to finish")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == 32:  # SPACE
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ STEP 1: detect face locations
        face_locations = face_recognition.face_locations(rgb)

        if len(face_locations) == 0:
            print("No face detected")
            continue

        # ✅ STEP 2: encode faces using locations
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        if len(face_encs) > 0:
            encodings.append(face_encs[0])
            print("Face captured")

cap.release()
cv2.destroyAllWindows()

# Save encodings
data = []
os.makedirs("data", exist_ok=True)

if os.path.exists("data/encodings.pkl"):
    with open("data/encodings.pkl", "rb") as f:
        data = pickle.load(f)

data.append({"name": name, "encodings": encodings})

with open("data/encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Registration completed")
