import pandas as pd
import matplotlib.pyplot as plt

file_path = "movement/timeseries/090_RelaxedTask_LeftWrist.txt"

df = pd.read_csv(file_path, header=None)

print(df.head())
print(df.shape)

plt.figure(figsize=(14,6))

plt.plot(df[1], label="Accel X")
plt.plot(df[2], label="Accel Y")
plt.plot(df[3], label="Accel Z")

plt.title("Resting Task Wrist Accelerometry")
plt.xlabel("Timepoints")
plt.ylabel("Acceleration")

plt.legend()

plt.savefig("multichannel_signal.png")
print("Saved multichannel figure")