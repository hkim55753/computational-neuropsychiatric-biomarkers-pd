import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os

# -------------------------
# Load label function
# -------------------------
def get_label(subject_id):
    try:
        path = f"patients/patient_{subject_id}.json"
        with open(path, "r") as f:
            data = json.load(f)
        return data["condition"]
    except:
        return None

# -------------------------
# Get files
# -------------------------
files = glob.glob("movement/timeseries/*.txt")

pd_file = None
hc_file = None

# Find one PD and one Healthy file
for f in files:
    filename = os.path.basename(f)
    subject_id = filename.split("_")[0]
    label = get_label(subject_id)

    if label == "Parkinson's" and pd_file is None:
        pd_file = f
    if label == "Healthy" and hc_file is None:
        hc_file = f

    if pd_file and hc_file:
        break

# -------------------------
# Load data
# -------------------------
def load_signal(file):
    df = pd.read_csv(file, header=None)
    return df

pd_df = load_signal(pd_file)
hc_df = load_signal(hc_file)

# -------------------------
# Plot accelerometer X (column 1)
# -------------------------
plt.figure(figsize=(12,5))

plt.plot(pd_df[1].values, label="Parkinson's", alpha=0.8)
plt.plot(hc_df[1].values, label="Healthy", alpha=0.8)

plt.title("Figure 2: Accelerometer X Signal (PD vs Healthy)")
plt.xlabel("Timepoints")
plt.ylabel("Acceleration")
plt.legend()

plt.tight_layout()
plt.savefig("fig2_signal_plot.png", dpi=300)
plt.show()

print("Saved: fig2_signal_plot.png")