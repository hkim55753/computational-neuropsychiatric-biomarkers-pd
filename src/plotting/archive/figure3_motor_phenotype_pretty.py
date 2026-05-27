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
# LOAD
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()

# =========================================================
# FEATURES
# =========================================================
def extract(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32).reshape(-1, 976)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    var = np.mean(np.var(x, axis=1))

    dx = np.diff(x, axis=1)
    jerk = np.mean(np.var(dx, axis=1))

    fft = np.fft.rfft(x, axis=1)
    power = np.abs(fft) ** 2
    p = power / (np.sum(power, axis=1, keepdims=True) + 1e-12)
    entropy = -np.mean(np.sum(p * np.log(p + 1e-12), axis=1))

    return var, entropy, jerk


# =========================================================
# COMPUTE
# =========================================================
rows = []

for _, r in df.iterrows():
    try:
        v, e, j = extract(r["id"])
        rows.append([r["id"], r["label"], v, e, j])
    except:
        continue

data = pd.DataFrame(rows, columns=["id", "label", "var", "entropy", "jerk"])

hc = data[data["label"] == 0]
pdg = data[data["label"] == 1]

features = ["var", "entropy", "jerk"]
titles = ["Variability", "Entropy", "Jerkiness"]

# =========================================================
# PLOT (CLEAN PUBLICATION STYLE)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for i, f in enumerate(features):
    ax = axes[i]

    hc_vals = hc[f].values
    pd_vals = pdg[f].values

    # jitter
    x1 = np.random.normal(0, 0.04, len(hc_vals))
    x2 = np.random.normal(1, 0.04, len(pd_vals))

    ax.scatter(x1, hc_vals, alpha=0.5, label="Healthy")
    ax.scatter(x2, pd_vals, alpha=0.5, label="Parkinson's")

    # means
    ax.plot([0, 1], [hc_vals.mean(), pd_vals.mean()], color="black", linewidth=2)

    # significance
    _, p = mannwhitneyu(hc_vals, pd_vals)

    ax.set_title(f"{titles[i]}\np = {p:.3g}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HC", "PD"])

    if i == 0:
        ax.set_ylabel("Feature Value")

plt.suptitle("Figure 3 — Motor Phenotype in Parkinson’s Disease", y=1.05)
plt.tight_layout()

plt.savefig("figures/figure3_motor_phenotype_pretty.png", dpi=300)
plt.close()

print("Saved → figures/figure3_motor_phenotype_pretty.png")