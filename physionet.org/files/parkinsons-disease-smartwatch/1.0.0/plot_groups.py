import pandas as pd
import matplotlib.pyplot as plt

# Load your extracted dataset
df = pd.read_csv("features.csv")

# Clean missing labels (important)
df = df.dropna(subset=["label"])

# Keep only PD and Healthy (simplify first analysis)
df = df[df["label"].isin(["Parkinson's disease", "Healthy"])]

# Separate groups
pd_group = df[df["label"] == "Parkinson's disease"]["accel_x_std"]
hc_group = df[df["label"] == "Healthy"]["accel_x_std"]

# Plot
plt.figure(figsize=(6,5))

plt.boxplot([pd_group, hc_group], tick_labels=["PD", "Healthy"])

plt.title("Movement Variability: PD vs Healthy")
plt.ylabel("Accel X Standard Deviation")

plt.savefig("pd_vs_healthy_boxplot.png", dpi=300)
print("Saved figure: pd_vs_healthy_boxplot.png")