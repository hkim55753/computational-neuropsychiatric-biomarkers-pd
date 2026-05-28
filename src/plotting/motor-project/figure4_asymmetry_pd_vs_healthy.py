import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# ROOT (YOU ARE ALREADY INSIDE 1.0.0)
# =========================================================
BASE = "."

file_list_path = os.path.join(BASE, "preprocessed", "file_list.csv")
movement_dir = os.path.join(BASE, "preprocessed", "movement")

os.makedirs("figures", exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(file_list_path)
df = df[df["label"].isin([0, 1])].copy()


# =========================================================
# FEATURE
# =========================================================
def compute_asymmetry(subject_id):
    file_path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(-1, 976)

    mid = data.shape[0] // 2
    left = data[:mid]
    right = data[mid:]

    return np.mean(np.var(left, axis=1)) - np.mean(np.var(right, axis=1))


# =========================================================
# COMPUTE FEATURES
# =========================================================
results = []

for _, row in df.iterrows():
    try:
        results.append([
            row["id"],
            row["label"],
            compute_asymmetry(row["id"])
        ])
    except Exception as e:
        print("skip", row["id"], e)

results = pd.DataFrame(results, columns=["id", "label", "asym"])


# =========================================================
# GROUPS
# =========================================================
hc = results[results["label"] == 0]["asym"]
pdg = results[results["label"] == 1]["asym"]

print("Healthy:", len(hc), "PD:", len(pdg))


# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(6, 5))
plt.violinplot([hc.values, pdg.values], showmeans=True)

plt.xticks([1, 2], ["Healthy", "Parkinson's"])
plt.ylabel("Left - Right Motor Power")
plt.title("Figure 4 — Bilateral Asymmetry (PD vs Healthy)")

plt.tight_layout()
plt.savefig("figures/figure4.png", dpi=300)
plt.close()

print("DONE → figures/figure4.png")