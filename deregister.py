# deregister.py
# Removes a registered face by name from encodings.pkl

import pickle
import os

DATA_PATH = "data/encodings.pkl"

if not os.path.exists(DATA_PATH):
    print("No registered faces found.")
    exit()

name_to_remove = input("Enter exact name to deregister: ").strip()

with open(DATA_PATH, "rb") as f:
    users = pickle.load(f)

updated_users = [u for u in users if u["name"] != name_to_remove]

if len(updated_users) == len(users):
    print("Face not found. No changes made.")
else:
    with open(DATA_PATH, "wb") as f:
        pickle.dump(updated_users, f)
    print(f"Face deregistered successfully: {name_to_remove}")
