import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

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
# TASK-AWARE FEATURE EXTRACTION
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

def extract_task_features(subject_id):
    path = os.path.join(movement_dir, f"{int(subject_id):03d}_ml.bin")

    x = np.fromfile(path, dtype=np.float32).reshape(-1, 976)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

    # split into pseudo-task segments (based on equal partitioning)
    n_tasks = len(TASKS)
    step = x.shape[0] // n_tasks

    features = []

    for i in range(n_tasks):
        segment = x[i * step:(i + 1) * step]

        if segment.shape[0] == 0:
            continue

        # core features per task
        var = np.mean(np.var(segment, axis=1))

        dx = np.diff(segment, axis=1)
        jerk = np.mean(np.var(dx, axis=1))

        fft = np.fft.rfft(segment, axis=1)
        power = np.abs(fft) ** 2
        p = power / (np.sum(power, axis=1, keepdims=True) + 1e-12)
        entropy = -np.mean(np.sum(p * np.log(p + 1e-12), axis=1))

        features.extend([var, entropy, jerk])

    return np.array(features)


# =========================================================
# BUILD DATASET
# =========================================================
X = []
y = []

for _, r in df.iterrows():
    try:
        feat = extract_task_features(r["id"])
        if len(feat) > 0:
            X.append(feat)
            y.append(r["label"])
    except:
        continue

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# =========================================================
# CLASSIFICATION
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(6, 6))

for train, test in cv.split(X, y):
    model = LogisticRegression(max_iter=2000)
    model.fit(X[train], y[train])

    prob = model.predict_proba(X[test])[:, 1]

    fpr, tpr, _ = roc_curve(y[test], prob)
    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tprs.append(tpr_interp)

    plt.plot(fpr, tpr, alpha=0.3)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color="black",
         label=f"Task-aware ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
         linewidth=2)

plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Figure 5 — Task-Aware PD Classification")
plt.legend()

plt.tight_layout()
plt.savefig("figures/figure5_task_aware_roc.png", dpi=300)
plt.close()

print("AUC:", mean_auc, "±", std_auc)
print("Saved → figures/figure5_task_aware_roc.png")