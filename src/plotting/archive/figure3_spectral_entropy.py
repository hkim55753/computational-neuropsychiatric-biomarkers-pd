import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

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
# LOAD DATA
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()

# =========================================================
# SAFE SPECTRAL ENTROPY
# =========================================================
def spectral_entropy(x):
    # FFT
    fft = np.fft.rfft(x)

    power = np.abs(fft) ** 2

    # normalize to probability distribution
    power = power + 1e-12
    p = power / np.sum(power)

    # entropy
    return -np.sum(p * np.log(p))


# =========================================================
# FEATURE EXTRACTION (ROBUST)
# =========================================================
def subject_entropy(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32)

    # safety check
    if len(x) == 0:
        return np.nan

    x = x.reshape(-1, 976)

    # normalize per subject
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # compute entropy per channel then average
    entropies = []

    for row in x:
        if np.std(row) < 1e-6:
            continue
        entropies.append(spectral_entropy(row))

    if len(entropies) == 0:
        return np.nan

    return np.mean(entropies)


# =========================================================
# COMPUTE FEATURES
# =========================================================
results = []

for _, r in df.iterrows():
    try:
        val = subject_entropy(r["id"])
        if not np.isnan(val):
            results.append([r["id"], r["label"], val])
    except:
        continue

results = pd.DataFrame(results, columns=["id", "label", "entropy"])

# remove any leftover NaNs
results = results.dropna()

# =========================================================
# GROUPS
# =========================================================
hc = results[results["label"] == 0]["entropy"]
pdg = results[results["label"] == 1]["entropy"]

print("HC n =", len(hc), "PD n =", len(pdg))

# =========================================================
# STATS
# =========================================================
stat, p = mannwhitneyu(hc, pdg, alternative="two-sided")

d = (pdg.mean() - hc.mean()) / np.sqrt((hc.std()**2 + pdg.std()**2) / 2)

print(f"p = {p:.3e}")
print(f"d = {d:.3f}")

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(6, 5))

plt.violinplot([hc.values, pdg.values], showmeans=True)

plt.xticks([1, 2], ["Healthy", "Parkinson's"])
plt.ylabel("Spectral Entropy")
plt.title("Figure 3 — Motor Signal Complexity (PD vs Healthy)")

plt.text(
    1.5,
    max(results["entropy"]) * 0.95,
    f"p = {p:.2e}\nd = {d:.2f}",
    ha="center"
)

plt.tight_layout()
plt.savefig("figures/figure3_spectral_entropy.png", dpi=300)
plt.close()

print("Saved → figures/figure3_spectral_entropy.png")