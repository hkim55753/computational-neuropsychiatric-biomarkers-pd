import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
from utils.constants import movement_dir, patient_dir
from utils.data_handling import load_all_files, get_data_from_txt_file


# -----------------------------
# CONFIG
# -----------------------------
SUBJECT_ID = "060"
TASK_FILTER = "HoldWeight"
SENSOR_FILTER = "Accelerometer"

RAW_CHANNELS_TO_PLOT = ["X", "Y", "Z"]


# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.concat(load_all_files(movement_dir))

df = df[df["subject_id"] == SUBJECT_ID].reset_index(drop=True)
df = df[df["file_name"].str.contains(TASK_FILTER)].reset_index(drop=True)


# -----------------------------
# LOAD RAW SIGNALS (LEFT + RIGHT)
# -----------------------------
def load_subject_signal(row):
    file_name = row["file_name"]
    file_path = movement_dir + file_name

    channels = row["channels"]
    n_channels = len(channels)

    data = get_data_from_txt_file(file_path, n_channels)

    # filter accelerometer channels
    idxs = [
        i for i, ch in enumerate(channels)
        if SENSOR_FILTER in ch
    ]

    data = data[:, idxs]
    return data


left = load_subject_signal(df.loc[0])
right = load_subject_signal(df.loc[1])


# -----------------------------
# ALIGN DIMENSIONS SAFELY
# -----------------------------
min_len = min(left.shape[0], right.shape[0])
left = left[:min_len]
right = right[:min_len]


# -----------------------------
# PLOT FIGURE 2
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

time = np.arange(min_len)


# ---- RAW LEFT ----
axes[0, 0].plot(left[:, :3])
axes[0, 0].set_title("Raw Signal — Left Wrist")
axes[0, 0].set_ylabel("Acceleration (g)")
axes[0, 0].legend(["X", "Y", "Z"])
axes[0, 0].set_ylim([-0.3, 0.3])


# ---- RAW RIGHT ----
axes[0, 1].plot(right[:, :3])
axes[0, 1].set_title("Raw Signal — Right Wrist")
axes[0, 1].legend(["X", "Y", "Z"])
axes[0, 1].set_ylim([-0.3, 0.3])


# -----------------------------
# LOAD PROCESSED DATA (.bin)
# -----------------------------
preprocessed_dir = "../preprocessed/movement/"
bin_path = f"{preprocessed_dir}{SUBJECT_ID}_ml.bin"

processed = np.fromfile(bin_path, dtype=np.float32)

# infer shape (same as your processed pipeline assumption)
processed = processed.reshape(-1, 976)

# pick accelerometer-like subset heuristically (first 6 channels shown here)
proc_left = processed[:3, :].T
proc_right = processed[3:6, :].T


# ---- PROCESSED LEFT ----
axes[1, 0].plot(proc_left)
axes[1, 0].set_title("Processed Signal — Left Wrist")
axes[1, 0].set_ylabel("Detrended Acceleration")
axes[1, 0].legend(["X", "Y", "Z"])
axes[1, 0].set_ylim([-0.15, 0.15])


# ---- PROCESSED RIGHT ----
axes[1, 1].plot(proc_right)
axes[1, 1].set_title("Processed Signal — Right Wrist")
axes[1, 1].legend(["X", "Y", "Z"])
axes[1, 1].set_ylim([-0.15, 0.15])

# -----------------------------
# FINAL FORMATTING
# -----------------------------
os.makedirs("figures", exist_ok=True)

plt.suptitle(f"Figure 2 — Raw vs Processed Motor Signals (Subject {SUBJECT_ID}, {TASK_FILTER})")
plt.tight_layout()

plt.savefig(f"figures/figure2_raw_vs_processed_{SUBJECT_ID}.png", dpi=300)
plt.close()

print("Figure 2 saved successfully.")