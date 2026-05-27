import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

# =========================================================
# STYLE (Nature Communications vibe)
# =========================================================
sns.set(style="whitegrid", context="paper", font_scale=1.2)

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
# FEATURES (UNCHANGED)
# =========================================================
def extract(subject_id):
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
# BUILD DATAFRAME
# =========================================================
rows = []

for _, r in df.iterrows():
    try:
        v, e, j = extract(r["id"])
        rows.append([r["id"], r["label"], v, e, j])
    except:
        continue

data = pd.DataFrame(rows, columns=["id", "label", "var", "entropy", "jerk"])

data["group"] = data["label"].map({0: "Healthy", 1: "Parkinson's"})

# melt for seaborn
long_df = data.melt(
    id_vars=["id", "group"],
    value_vars=["var", "entropy", "jerk"],
    var_name="feature",
    value_name="value"
)

# pretty labels
label_map = {
    "var": "Variability",
    "entropy": "Entropy",
    "jerk": "Jerkiness"
}
long_df["feature"] = long_df["feature"].map(label_map)

# =========================================================
# STATS (still used for annotation)
# =========================================================
stats = {}
for f in ["Variability", "Entropy", "Jerkiness"]:
    hc = data[data["group"] == "Healthy"][f.lower() if f != "Jerkiness" else "jerk"]
    pdg = data[data["group"] == "Parkinson's"][f.lower() if f != "Jerkiness" else "jerk"]

    _, p = mannwhitneyu(hc, pdg)
    stats[f] = p


# =========================================================
# PLOT (NATURE COMMUNICATIONS STYLE)
# =========================================================
plt.figure(figsize=(10, 4))

ax = sns.violinplot(
    data=long_df,
    x="feature",
    y="value",
    hue="group",
    split=True,
    inner=None,
    palette=["#4C72B0", "#DD8452"]
)

sns.stripplot(
    data=long_df,
    x="feature",
    y="value",
    hue="group",
    dodge=True,
    alpha=0.35,
    size=3,
    palette=["#4C72B0", "#DD8452"],
    ax=ax
)

# clean legend duplication
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Group")

# annotate p-values
ymax = long_df["value"].max()

for i, f in enumerate(["Variability", "Entropy", "Jerkiness"]):
    ax.text(
        i,
        ymax * 1.05,
        f"p={stats[f]:.2g}",
        ha="center",
        fontsize=10
    )

ax.set_title("Figure 3 — Motor Phenotype in Parkinson’s Disease", pad=20)
ax.set_xlabel("")
ax.set_ylabel("Normalized Feature Value")

plt.tight_layout()
plt.savefig("figures/figure3_motor_phenotype_nature.png", dpi=300)
plt.close()

print("Saved → figures/figure3_motor_phenotype_nature.png")