import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PATHS (cross-workspace safe)
# -----------------------------
ANALYSIS_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd")
DATA_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/data/raw/1.0.0")

MOV_DIR = DATA_DIR / "preprocessed" / "movement"
PREPROCESSED = ANALYSIS_DIR / "preprocessed"
OUT_DIR = ANALYSIS_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_LIST = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/data/raw/1.0.0/preprocessed/file_list.csv")

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(FILE_LIST)

# pick representative subjects
HC_ID = 78
PD_ID = 60

# -----------------------------
# LOAD SIGNAL
# -----------------------------
def load_subject(sub_id):
    path = MOV_DIR / f"{sub_id:03d}_ml.bin"
    x = np.fromfile(path, dtype=np.float32)

    # robust reshape
    for d in [976, 600, 512, 300]:
        if x.size % d == 0:
            return x.reshape((-1, d))

    raise ValueError(f"Cannot reshape {sub_id}, size={x.size}")

hc = load_subject(HC_ID)
pd = load_subject(PD_ID)

# -----------------------------
# SIMPLE FEATURE: signal variability
# -----------------------------
hc_var = np.std(hc, axis=1)
pd_var = np.std(pd, axis=1)

# -----------------------------
# FIGURE 2
# -----------------------------
fig = plt.figure(figsize=(12, 7))

# ---------------- Panel A: raw traces ----------------
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

t = np.arange(300)

ax1.plot(hc[0, :300], label="HC", alpha=0.7)
ax1.plot(pd[0, :300], label="PD", alpha=0.7)

ax1.set_title("A. Raw Wrist Acceleration (HoldWeight)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Signal amplitude")
ax1.legend()

# ---------------- Panel B: zoom ----------------
ax2 = plt.subplot2grid((2, 2), (1, 0))

ax2.plot(pd[0, 100:200])
ax2.set_title("B. PD Tremor Segment (Zoom)")
ax2.set_xlabel("Time (samples)")
ax2.set_ylabel("Amplitude")

# ---------------- Panel C: variability ----------------
ax3 = plt.subplot2grid((2, 2), (1, 1))

ax3.hist(hc_var, bins=30, alpha=0.5, label="HC")
ax3.hist(pd_var, bins=30, alpha=0.5, label="PD")

ax3.set_title("C. Motor Variability Distribution")
ax3.set_xlabel("Signal Variability (std)")
ax3.set_ylabel("Number of channels")
ax3.legend()

plt.suptitle("Figure 2: Motor Signatures from Wearable Sensors")

plt.tight_layout()

out_path = OUT_DIR / "fig2_motor_signatures.png"
plt.savefig(out_path, dpi=300)

print(f"Saved Figure 2 → {out_path}")