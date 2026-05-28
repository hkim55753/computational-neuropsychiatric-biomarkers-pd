import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/data/raw/1.0.0")
MOV_DIR = DATA_DIR / "preprocessed" / "movement"
Q_DIR = DATA_DIR / "preprocessed" / "questionnaire"
FILE_LIST = DATA_DIR / "preprocessed" / "file_list.csv"

OUT_DIR = Path("/workspaces/computational-neuropsychiatric-biomarkers-pd/outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD META
# -----------------------------
df = pd.read_csv(FILE_LIST)

if "label" not in df.columns:
    df["label"] = df["condition"].map({
        "Healthy": 0,
        "Parkinson's": 1
    })

# -----------------------------
# LOAD DATA
# -----------------------------
def load_subject(sub_id):
    path = MOV_DIR / f"{sub_id:03d}_ml.bin"
    x = np.fromfile(path, dtype=np.float32)

    for d in [976, 600, 512, 300]:
        if x.size % d == 0:
            return x.reshape((-1, d))
    return None

def load_questionnaire(sub_id):
    path = Q_DIR / f"{sub_id:03d}_ml.bin"
    try:
        return np.fromfile(path, dtype=np.float32)
    except:
        return None

# -----------------------------
# FEATURES
# -----------------------------
def motor_index(x):
    # composite instability metric (more robust than single std)
    return np.mean(np.std(x, axis=1)) + 0.5 * np.mean(np.sqrt(np.mean(x**2, axis=1)))

def sleep_index(q):
    if q is None:
        return None
    return np.mean(q)

# -----------------------------
# BUILD SPACE
# -----------------------------
X_sleep = []
X_motor = []
y = []

for _, row in df.iterrows():
    sid = int(row["id"])

    x = load_subject(sid)
    q = load_questionnaire(sid)

    if x is None or q is None:
        continue

    s = sleep_index(q)
    m = motor_index(x)

    if np.isnan(s):
        continue

    X_sleep.append(s)
    X_motor.append(m)
    y.append(row["label"])

X_sleep = np.array(X_sleep)
X_motor = np.array(X_motor)
y = np.array(y)

# normalize (important for clean decision boundary)
X_sleep = (X_sleep - X_sleep.mean()) / (X_sleep.std() + 1e-8)
X_motor = (X_motor - X_motor.mean()) / (X_motor.std() + 1e-8)

X = np.column_stack([X_sleep, X_motor])

# -----------------------------
# CLASSIFIER (phenotype boundary)
# -----------------------------
clf = LogisticRegression()
clf.fit(X, y)

# mesh grid for decision surface
x_min, x_max = X_sleep.min() - 0.5, X_sleep.max() + 0.5
y_min, y_max = X_motor.min() - 0.5, X_motor.max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

# -----------------------------
# FIGURE
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# decision surface
contour = ax.contourf(xx, yy, probs, levels=20, cmap="coolwarm", alpha=0.4)

# scatter
ax.scatter(
    X_sleep[y == 0],
    X_motor[y == 0],
    label="HC",
    alpha=0.8
)

ax.scatter(
    X_sleep[y == 1],
    X_motor[y == 1],
    label="PD",
    alpha=0.8
)

ax.set_xlabel("Sleep symptom burden (z-score)")
ax.set_ylabel("Motor instability index (z-score)")
ax.set_title("Figure 5: Sleep–Motor Phenotype Space")

ax.legend()

cbar = plt.colorbar(contour)
cbar.set_label("PD probability")

plt.tight_layout()

out_path = OUT_DIR / "fig5_sleep_motor_space.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"Saved Figure 5 → {out_path}")