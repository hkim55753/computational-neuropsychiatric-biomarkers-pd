import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/data/raw/1.0.0")
MOV_DIR = DATA_DIR / "preprocessed" / "movement"
FILE_LIST = DATA_DIR / "preprocessed" / "file_list.csv"

OUT_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(FILE_LIST)

if "label" not in df.columns:
    df["label"] = df["condition"].map({
        "Healthy": 0,
        "Parkinson's": 1
    })

# -----------------------------
# SAFE LOADER
# -----------------------------
def load_subject(sub_id):
    path = MOV_DIR / f"{sub_id:03d}_ml.bin"

    try:
        x = np.fromfile(path, dtype=np.float32)
    except:
        return None

    if len(x) < 100:
        return None

    for d in [976, 600, 512, 300]:
        if x.size % d == 0:
            return x.reshape((-1, d))

    return None

# -----------------------------
# FEATURES
# -----------------------------
def entropy(signal):
    hist, _ = np.histogram(signal, bins=20, density=True)
    hist = hist + 1e-8
    return -np.sum(hist * np.log(hist))

features = []
labels = []

for _, row in df.iterrows():
    sid = int(row["id"])
    x = load_subject(sid)

    if x is None:
        continue

    try:
        std_feat = np.mean(np.std(x, axis=1))
        rms_feat = np.mean(np.sqrt(np.mean(x**2, axis=1)))
        ent_feat = np.mean([entropy(ch) for ch in x])

        features.append([std_feat, rms_feat, ent_feat])
        labels.append(row["label"])

    except:
        continue

features = np.array(features)
labels = np.array(labels)

if features.ndim != 2 or len(features) == 0:
    raise ValueError("Empty feature matrix — check data loading.")

hc = features[labels == 0]
pd = features[labels == 1]

# -----------------------------
# FIGURE 4 (2 PANELS ONLY)
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# ---------------- Panel A: variability ----------------
ax[0].boxplot([hc[:, 0], pd[:, 0]], labels=["HC", "PD"])
ax[0].set_title("A. Signal Variability (Std)")
ax[0].set_ylabel("Value")

# ---------------- Panel B: movement energy ----------------
ax[1].boxplot([hc[:, 1], pd[:, 1]], labels=["HC", "PD"])
ax[1].set_title("B. Movement Energy (RMS)")
ax[1].set_ylabel("Value")

plt.suptitle("Figure 4: Wearable-Derived Motor Biomarkers")

plt.tight_layout()

out_path = OUT_DIR / "fig4_motor_features.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"Saved Figure 4 → {out_path}")