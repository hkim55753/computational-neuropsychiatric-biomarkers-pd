import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
ANALYSIS_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd")
DATA_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/data/raw/1.0.0")

Q_DIR = DATA_DIR / "preprocessed" / "questionnaire"
OUT_DIR = ANALYSIS_DIR / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_LIST = DATA_DIR / "preprocessed" / "file_list.csv"

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(FILE_LIST)

if "label" not in df.columns:
    df["label"] = df["condition"].map({
        "Healthy": 0,
        "Parkinson's": 1
    })

# -----------------------------
# SLEEP ITEMS
# -----------------------------
sleep_idx = {
    21: "Daytime sleepiness",
    22: "Insomnia",
    23: "Intense vivid dreams",
    24: "REM behavior",
    25: "Restless legs",
    9: "Nocturia"
}

sleep_names = list(sleep_idx.values())

# -----------------------------
# LOAD QUESTIONNAIRES
# -----------------------------
def load_questionnaire(sub_id):
    path = Q_DIR / f"{sub_id:03d}_ml.bin"
    return np.fromfile(path, dtype=np.float32)

sleep_scores = []
labels = []

for _, row in df.iterrows():
    sid = int(row["id"])
    try:
        q = np.nan_to_num(load_questionnaire(sid))

        vals = [q[idx] for idx in sleep_idx.keys() if idx < len(q)]
        if len(vals) == 0:
            continue

        score = np.mean(vals)

        sleep_scores.append(score)
        labels.append(row["label"])

    except:
        continue

sleep_scores = np.array(sleep_scores)
labels = np.array(labels)

# z-score normalization
sleep_scores = (sleep_scores - np.mean(sleep_scores)) / (np.std(sleep_scores) + 1e-8)

hc = sleep_scores[labels == 0]
pd = sleep_scores[labels == 1]

# -----------------------------
# FIGURE
# -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# ---------------- PANEL A (FIXED OVERLAP) ----------------
ax[0].hist(
    hc,
    bins=25,
    alpha=0.7,
    label="HC",
    density=True,
    histtype="step",
    linewidth=2,
    color="blue"
)

ax[0].hist(
    pd,
    bins=25,
    alpha=0.7,
    label="PD",
    density=True,
    histtype="step",
    linewidth=2,
    color="orange"
)

ax[0].set_title("A. Sleep Symptom Burden")
ax[0].set_xlabel("Sleep burden (z-score)")
ax[0].set_ylabel("Density")
ax[0].legend()

# ---------------- PANEL B ----------------
sleep_matrix = np.zeros((len(df), len(sleep_idx)))

for i, (_, row) in enumerate(df.iterrows()):
    sid = int(row["id"])
    try:
        q = np.nan_to_num(load_questionnaire(sid))
        for j, idx in enumerate(sleep_idx.keys()):
            if idx < len(q):
                sleep_matrix[i, j] = q[idx]
    except:
        continue

hc_mean = sleep_matrix[df["label"] == 0].mean(axis=0)
pd_mean = sleep_matrix[df["label"] == 1].mean(axis=0)

x = np.arange(len(sleep_names))

ax[1].bar(x - 0.2, hc_mean, width=0.4, label="HC", color="blue")
ax[1].bar(x + 0.2, pd_mean, width=0.4, label="PD", color="orange")

ax[1].set_xticks(x)
ax[1].set_xticklabels(sleep_names, rotation=45, ha="right")
ax[1].set_title("B. Individual Sleep Symptoms")
ax[1].set_ylabel("Mean severity")
ax[1].legend()

# ---------------- PANEL C (FIXED CUT-OFF) ----------------
ax[2].boxplot(
    [hc, pd],
    labels=["HC", "PD"],
    patch_artist=True,
    boxprops=dict(facecolor="lightgray")
)

ax[2].set_title("C. Sleep Burden Distribution")
ax[2].set_ylabel("Sleep burden (z-score)")

# ---------------- LAYOUT FIX (IMPORTANT) ----------------
plt.suptitle("Figure 3: Sleep-Related Symptom Burden in Parkinsonism")

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# ---------------- SAVE ----------------
out_path = OUT_DIR / "fig3_sleep_phenotype.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"Saved Figure 3 → {out_path}")