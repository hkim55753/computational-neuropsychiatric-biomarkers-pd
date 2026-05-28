import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
# TASK STRUCTURE (same as Fig 5)
# =========================================================
TASKS = [
    "Relaxed",
    "StretchHold",
    "LiftHold",
    "HoldWeight",
    "DrinkGlas",
    "CrossArms",
    "TouchNose",
    "Entrainment"
]

# =========================================================
# FEATURE EXTRACTION PER TASK
# =========================================================
def extract_task_matrix(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32).reshape(-1, 976)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    n_tasks = len(TASKS)
    step = x.shape[0] // n_tasks

    feature_vector = []

    for i in range(n_tasks):
        seg = x[i * step:(i + 1) * step]

        if seg.shape[0] == 0:
            feature_vector.extend([0, 0, 0])
            continue

        var = np.mean(np.var(seg, axis=1))

        dx = np.diff(seg, axis=1)
        jerk = np.mean(np.var(dx, axis=1))

        fft = np.fft.rfft(seg, axis=1)
        power = np.abs(fft) ** 2
        p = power / (np.sum(power, axis=1, keepdims=True) + 1e-12)
        entropy = -np.mean(np.sum(p * np.log(p + 1e-12), axis=1))

        feature_vector.extend([var, entropy, jerk])

    return np.array(feature_vector)

# =========================================================
# BUILD DATASET
# =========================================================
X = []
y = []

for _, r in df.iterrows():
    try:
        X.append(extract_task_matrix(r["id"]))
        y.append(r["label"])
    except:
        continue

X = np.array(X)
y = np.array(y)

# normalize (important for coefficients)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# =========================================================
# TRAIN MODEL
# =========================================================
model = LogisticRegression(max_iter=5000)
model.fit(Xn, y)

coef = model.coef_[0]

# =========================================================
# ORGANIZE COEFFICIENTS
# =========================================================
features = ["var", "entropy", "jerk"]

task_names = []
vals = []

for i, task in enumerate(TASKS):
    for j, feat in enumerate(features):
        task_names.append(f"{task}\n{feat}")
        vals.append(coef[i * 3 + j])

vals = np.array(vals)

# sort by absolute importance
idx = np.argsort(np.abs(vals))[::-1]
task_names = np.array(task_names)[idx]
vals = vals[idx]

# =========================================================
# PLOT FIGURE 6
# =========================================================
plt.figure(figsize=(10, 5))

colors = ["#4C72B0" if v < 0 else "#DD8452" for v in vals]

plt.bar(range(len(vals)), vals, color=colors)

plt.xticks(range(len(vals)), task_names, rotation=90)
plt.ylabel("Model Coefficient (PD direction)")
plt.title("Figure 6 — Task-Specific Motor Biomarker Contributions")

plt.axhline(0, color="black", linewidth=1)

plt.tight_layout()
plt.savefig("figures/figure6_task_feature_importance.png", dpi=300)
plt.close()

print("Saved → figures/figure6_task_feature_importance.png")