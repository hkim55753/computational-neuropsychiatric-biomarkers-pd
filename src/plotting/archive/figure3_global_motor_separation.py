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
# LOAD LABELS
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()


# =========================================================
# FEATURE: GLOBAL MOTOR ACTIVITY (ROBUST)
# =========================================================
def motor_activity(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32)
    x = x.reshape(-1, 976)

    # normalize per subject (important)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # global motor activity = variance + energy combo
    var = np.mean(np.var(x, axis=1))
    rms = np.sqrt(np.mean(x ** 2))

    return 0.5 * var + 0.5 * rms


# =========================================================
# COMPUTE FEATURES
# =========================================================
results = []

for _, r in df.iterrows():
    try:
        val = motor_activity(r["id"])
        results.append([r["id"], r["label"], val])
    except:
        continue

results = pd.DataFrame(results, columns=["id", "label", "motor"])


# =========================================================
# GROUPS
# =========================================================
hc = results[results["label"] == 0]["motor"]
pdg = results[results["label"] == 1]["motor"]


# =========================================================
# STATISTICS
# =========================================================
stat, p = mannwhitneyu(hc, pdg, alternative="two-sided")

pooled_std = np.sqrt((hc.std()**2 + pdg.std()**2) / 2)
effect_size = (pdg.mean() - hc.mean()) / pooled_std


print(f"Mann-Whitney p = {p:.3e}")
print(f"Cohen's d = {effect_size:.3f}")


# =========================================================
# PLOT FIGURE 3 (CLEAN + STRONG)
# =========================================================
plt.figure(figsize=(6, 5))

plt.violinplot([hc.values, pdg.values], showmeans=True)

plt.xticks([1, 2], ["Healthy", "Parkinson's"])
plt.ylabel("Global Motor Activity")
plt.title("Figure 3 — Motor Activity Separation")

# annotate stats
plt.text(
    1.5,
    max(results["motor"]) * 0.95,
    f"p = {p:.2e}\nd = {effect_size:.2f}",
    ha="center"
)

plt.tight_layout()

plt.savefig("figures/figure3_motor_separation.png", dpi=300)
plt.close()

print("Saved → figures/figure3_motor_separation.png")