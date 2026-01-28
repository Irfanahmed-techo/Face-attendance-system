# Face Authentication Attendance System

## Overview
This project is a real-time **face recognition–based attendance system** developed using Python. It supports face registration, live face recognition through a webcam, punch-in/punch-out attendance marking, and basic spoof (liveness) detection.



## Features
- Face registration using webcam
- Real-time face recognition
- Punch-In / Punch-Out attendance logic
- CSV-based attendance storage
- Basic anti-spoof detection using motion analysis
- Supports multiple users



## Tech Stack
- Python 3.10
- OpenCV
- face_recognition (dlib)
- NumPy
- pandas



## Project Structure
face-attendance/
│
├── register.py          # Face registration using webcam
├── attendance.py        # Face recognition & attendance marking
├── spoof_check.py       # Basic anti-spoof (liveness) detection
├── README.md            # Project documentation
│
└── data/
    ├── encodings.pkl    # Stored face encodings (generated after registration)
    └── attendance.csv   # Attendance records (auto-generated)
