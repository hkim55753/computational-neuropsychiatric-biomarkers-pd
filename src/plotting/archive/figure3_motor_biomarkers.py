import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# PATHS (safe relative root)
# =========================================================
BASE = "."
file_list_path = os.path.join(
    BASE,
    "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv"
)
movement_dir = os.path.join(
    BASE,
    "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/movement"
)

os.makedirs("figures", exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()


# =========================================================
# FEATURE ENGINEERING (REAL SIGNAL BIOMARKERS)
# =========================================================
def extract_features(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32)
    x = x.reshape(-1, 976)

    # normalize per subject
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # 1. variability
    var = np.mean(np.var(x, axis=1))

    # 2. RMS energy
    rms = np.sqrt(np.mean(x ** 2))

    # 3. signal "jerkiness" (temporal derivative)
    dx = np.diff(x, axis=1)
    jerk = np.mean(np.var(dx, axis=1))

    # 4. peakiness (proxy for tremor bursts)
    peak = np.percentile(np.abs(x), 95)

    return var, rms, jerk, peak


# =========================================================
# BUILD DATASET
# =========================================================
rows = []

for _, r in df.iterrows():
    try:
        v, rms, j, p = extract_features(r["id"])
        rows.append([r["id"], r["label"], v, rms, j, p])
    except:
        continue

data = pd.DataFrame(
    rows,
    columns=["id", "label", "var", "rms", "jerk", "peak"]
)


# =========================================================
# GROUPS
# =========================================================
hc = data[data["label"] == 0]
pdg = data[data["label"] == 1]


# =========================================================
# PLOT (MULTI-FEATURE BIOMARKER MAP)
# =========================================================
features = ["var", "rms", "jerk", "peak"]

plt.figure(figsize=(10, 6))

x = np.arange(len(features))

hc_means = hc[features].mean().values
pd_means = pdg[features].mean().values

plt.plot(x, hc_means, marker="o", label="Healthy")
plt.plot(x, pd_means, marker="o", label="Parkinson's")

plt.xticks(x, features)
plt.ylabel("Normalized Feature Value")
plt.title("Figure 3 — Motor Biomarker Signature (PD vs Healthy)")
plt.legend()

plt.tight_layout()
plt.savefig("figures/figure3_biomarker_signature.png", dpi=300)
plt.close()

print("Saved → figures/figure3_biomarker_signature.png")