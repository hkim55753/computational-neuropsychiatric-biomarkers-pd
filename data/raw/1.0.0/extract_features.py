import pandas as pd
import numpy as np
import glob
import os
import json

# =========================
# Load all files
# =========================
files = glob.glob("movement/timeseries/*.txt")

results = []

# =========================
# Label function (FIXED)
# =========================
def get_label(subject_id):
    try:
        path = f"patients/patient_{subject_id}.json"
        with open(path, "r") as f:
            data = json.load(f)

        raw = str(data.get("condition", "")).lower()

        if "parkinson" in raw or "pd" == raw:
            return "Parkinson's"
        elif "healthy" in raw:
            return "Healthy"
        else:
            return "Other Movement Disorders"

    except:
        return None


# =========================
# Feature extraction loop
# =========================
for file in files:

    try:
        df = pd.read_csv(file, header=None)

        # ensure numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        filename = os.path.basename(file)

        # subject id extraction (robust)
        subject_id = filename.split("_")[0].lstrip("0")

        label = get_label(subject_id)

        # skip if no label
        if label is None:
            continue

        # skip broken files
        if df.shape[1] < 7:
            continue

        # =========================
        # SENSOR COLUMNS
        # =========================
        accel_x = df[1]
        accel_y = df[2]
        accel_z = df[3]

        gyro_x = df[4]
        gyro_y = df[5]
        gyro_z = df[6]

        # =========================
        # FEATURES
        # =========================

        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        results.append({
            "subject_id": subject_id,
            "label": label,
            "file": filename,

            # accelerometer variability
            "accel_x_std": np.nanstd(accel_x),
            "accel_y_std": np.nanstd(accel_y),
            "accel_z_std": np.nanstd(accel_z),

            # gyroscope variability
            "gyro_x_std": np.nanstd(gyro_x),
            "gyro_y_std": np.nanstd(gyro_y),
            "gyro_z_std": np.nanstd(gyro_z),

            # magnitude-based features (more clinically meaningful)
            "accel_mag_mean": np.nanmean(accel_mag),
            "accel_mag_std": np.nanstd(accel_mag),

            "gyro_mag_mean": np.nanmean(gyro_mag),
            "gyro_mag_std": np.nanstd(gyro_mag),
        })

    except Exception as e:
        print("Error processing:", file, "->", e)


# =========================
# Create dataframe
# =========================
features_df = pd.DataFrame(results)

print("\n=== Label distribution ===")
print(features_df["label"].value_counts())

print("\n=== Preview ===")
print(features_df.head())

# =========================
# Save
# =========================
features_df.to_csv("features.csv", index=False)

print("\nSaved features.csv successfully")