import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu

# =========================================================
# PATHS
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
# DATA
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()

# =========================================================
# FEATURES (ROBUST SET)
# =========================================================
def extract_features(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32).reshape(-1, 976)

    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # 1. variability
    var = np.mean(np.var(x, axis=1))

    # 2. entropy
    fft = np.fft.rfft(x, axis=1)
    power = np.abs(fft) ** 2
    p = power / (np.sum(power, axis=1, keepdims=True) + 1e-12)
    entropy = -np.mean(np.sum(p * np.log(p + 1e-12), axis=1))

    # 3. jerk (smoothness)
    dx = np.diff(x, axis=1)
    jerk = np.mean(np.var(dx, axis=1))

    return var, entropy, jerk


# =========================================================
# COMPUTE
# =========================================================
rows = []

for _, r in df.iterrows():
    try:
        v, e, j = extract_features(r["id"])
        rows.append([r["id"], r["label"], v, e, j])
    except:
        continue

data = pd.DataFrame(rows, columns=["id", "label", "var", "entropy", "jerk"])

# =========================================================
# GROUPS
# =========================================================
hc = data[data["label"] == 0]
pdg = data[data["label"] == 1]

features = ["var", "entropy", "jerk"]

# =========================================================
# STATISTICS TABLE
# =========================================================
stats = []

for f in features:
    h = hc[f]
    p = pdg[f]

    stat, pval = mannwhitneyu(h, p, alternative="two-sided")
    d = (p.mean() - h.mean()) / np.sqrt((h.std()**2 + p.std()**2)/2)

    stats.append([f, pval, d])

stats = pd.DataFrame(stats, columns=["feature", "p", "d"])

print(stats)

# =========================================================
# PLOT (MULTI-PANEL STYLE)
# =========================================================
plt.figure(figsize=(8, 5))

x = np.arange(len(features))

hc_means = [hc[f].mean() for f in features]
pd_means = [pdg[f].mean() for f in features]

plt.plot(x, hc_means, marker="o", label="Healthy")
plt.plot(x, pd_means, marker="o", label="Parkinson's")

plt.xticks(x, features)
plt.ylabel("Normalized Feature Value")
plt.title("Figure 3 — Multidimensional Motor Biomarker Signature")
plt.legend()

# annotate effect sizes
for i, row in stats.iterrows():
    plt.text(i, max(hc_means[i], pd_means[i]), f"d={row['d']:.2f}", ha="center")

plt.tight_layout()
plt.savefig("figures/figure3_multifeature.png", dpi=300)
plt.close()

print("Saved → figures/figure3_multifeature.png")