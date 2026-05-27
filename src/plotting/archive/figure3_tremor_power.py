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
# TREMOR FEATURE (KEY IMPROVEMENT)
# =========================================================
def tremor_power(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32)
    x = x.reshape(-1, 976)

    # normalize per subject
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # FFT along time axis
    fft = np.fft.rfft(x, axis=1)
    power = np.abs(fft) ** 2

    freqs = np.fft.rfftfreq(x.shape[1], d=1)

    # tremor band mask (approximate index range)
    band = (freqs >= 4) & (freqs <= 7)

    return np.mean(power[:, band])


# =========================================================
# COMPUTE FEATURES
# =========================================================
results = []

for _, r in df.iterrows():
    try:
        val = tremor_power(r["id"])
        results.append([r["id"], r["label"], val])
    except:
        continue

results = pd.DataFrame(results, columns=["id", "label", "tremor"])


# =========================================================
# GROUPS
# =========================================================
hc = results[results["label"] == 0]["tremor"]
pdg = results[results["label"] == 1]["tremor"]


# =========================================================
# STATS
# =========================================================
stat, p = mannwhitneyu(hc, pdg, alternative="two-sided")

d = (pdg.mean() - hc.mean()) / np.sqrt((hc.std()**2 + pdg.std()**2)/2)

print("p =", p)
print("d =", d)


# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(6, 5))

plt.violinplot([hc.values, pdg.values], showmeans=True)

plt.xticks([1, 2], ["Healthy", "Parkinson's"])
plt.ylabel("Tremor Band Power (4–7 Hz)")
plt.title("Figure 3 — Tremor Biomarker Separation")

plt.text(
    1.5,
    max(results["tremor"]) * 0.95,
    f"p = {p:.2e}\nd = {d:.2f}",
    ha="center"
)

plt.tight_layout()
plt.savefig("figures/figure3_tremor.png", dpi=300)
plt.close()

print("Saved → figures/figure3_tremor.png")