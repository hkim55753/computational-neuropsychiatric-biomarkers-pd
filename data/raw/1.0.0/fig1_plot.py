import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
df = pd.read_csv("features.csv")

print("\nLabel counts:")
print(df["label"].value_counts())

# =========================
# Choose feature
# =========================
feature = "accel_mag_std"

# =========================
# Filter groups
# =========================
pd_df = df[df["label"] == "Parkinson's"]
hc_df = df[df["label"] == "Healthy"]

print("\nSample sizes:")
print("PD:", len(pd_df))
print("Healthy:", len(hc_df))

# =========================
# Clean NaNs
# =========================
pd_vals = pd_df[feature].dropna()
hc_vals = hc_df[feature].dropna()

# =========================
# Plot
# =========================
plt.figure(figsize=(8,6))

data = [hc_vals, pd_vals]

plt.boxplot(data, labels=["Healthy", "Parkinson's"], showfliers=False)

# jittered points (important for research figures)
for i, group in enumerate(data):
    x = np.random.normal(i + 1, 0.04, size=len(group))
    plt.scatter(x, group, alpha=0.6, s=25)

plt.title("Motor Variability: Parkinson's vs Healthy")
plt.ylabel("Accelerometer Magnitude Std (g)")
plt.grid(alpha=0.3)
plt.savefig("fig1_pd_vs_healthy.png", dpi=300, bbox_inches="tight")
plt.show()
 
 

print(df["label"].value_counts())
print(df["label"].unique())