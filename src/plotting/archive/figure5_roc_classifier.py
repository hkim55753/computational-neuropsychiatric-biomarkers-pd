import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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
# FEATURE ENGINEERING (SAME AS FIG 3)
# =========================================================
def extract_features(subject_id):
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
# BUILD DATASET
# =========================================================
rows = []

for _, r in df.iterrows():
    try:
        v, e, j = extract_features(r["id"])
        rows.append([v, e, j, r["label"]])
    except:
        continue

data = pd.DataFrame(rows, columns=["var", "entropy", "jerk", "label"])

X = data[["var", "entropy", "jerk"]].values
y = data["label"].values

# =========================================================
# CROSS-VALIDATED ROC
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(6, 6))

for i, (train, test) in enumerate(cv.split(X, y)):
    model = LogisticRegression(max_iter=1000)
    model.fit(X[train], y[train])

    prob = model.predict_proba(X[test])[:, 1]

    fpr, tpr, _ = roc_curve(y[test], prob)
    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    plt.plot(fpr, tpr, alpha=0.3)

# mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color="black",
         label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
         linewidth=2)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Figure 5 — PD Classification from Wearable Biomarkers")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("figures/figure5_roc.png", dpi=300)
plt.close()

print(f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
print("Saved → figures/figure5_roc.png")