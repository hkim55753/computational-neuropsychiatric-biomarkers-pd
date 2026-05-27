import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = "."
file_list_path = os.path.join(BASE, "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/file_list.csv")
movement_dir = os.path.join(BASE, "physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/movement")

os.makedirs("figures", exist_ok=True)

df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()

tasks = [
    "Relaxed", "StretchHold", "LiftHold",
    "HoldWeight", "CrossArms",
    "TouchNose", "Entrainment"
]

# ---------------------------------------------------------
# REAL FEATURE: signal energy per task block
# ---------------------------------------------------------
def task_energy(subject_id):
    file_path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")
    x = np.fromfile(file_path, dtype=np.float32).reshape(-1, 976)

    # normalize per subject (IMPORTANT)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # split into pseudo task segments (approximation)
    n_tasks = len(tasks)
    splits = np.array_split(x, n_tasks)

    energies = [np.mean(s**2) for s in splits]
    return energies


# ---------------------------------------------------------
# BUILD DATA
# ---------------------------------------------------------
rows = []

for _, r in df.iterrows():
    try:
        energies = task_energy(r["id"])
        for t, e in zip(tasks, energies):
            rows.append([r["id"], r["label"], t, e])
    except:
        continue

data = pd.DataFrame(rows, columns=["id", "label", "task", "energy"])


# ---------------------------------------------------------
# AGGREGATE
# ---------------------------------------------------------
summary = data.groupby(["task", "label"])["energy"].mean().unstack()

summary = summary.reindex(tasks)

# ---------------------------------------------------------
# PLOT (CLEAN COMPARISON)
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(summary.index, summary[0], marker="o", label="Healthy")
plt.plot(summary.index, summary[1], marker="o", label="Parkinson's")

plt.xticks(rotation=45, ha="right")
plt.ylabel("Normalized Signal Energy")
plt.title("Figure 3 — Task-Specific Motor Energy Signatures")
plt.legend()

plt.tight_layout()
plt.savefig("figures/figure3_clean.png", dpi=300)
plt.close()

print("Saved → figures/figure3_clean.png")