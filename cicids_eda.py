import pandas as pd
import matplotlib.pyplot as plt

file_path = "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df["Flow Duration (seconds)"] = df["Flow Duration"] / 1_000_000

print("Dataset shape:", df.shape)

plt.hist(df["Flow Duration (seconds)"], bins=50)

plt.xlabel("Flow Duration (seconds)")
plt.ylabel("Number of Flows")
plt.title("Distribution of Flow Duration")

plt.show()