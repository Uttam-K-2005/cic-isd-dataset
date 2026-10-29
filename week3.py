import pandas as pd
import matplotlib.pyplot as plt

file_path = "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)
# print("\n--- DATASET INFO ---")
# df.info()
print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()

print(missing[missing > 0])
# print("\n--- STATISTICAL SUMMARY ---")
# print(df.describe())
import numpy as np

print("\n--- INFINITE VALUES ---")

infinite_values = np.isinf(df.select_dtypes(include="number")).sum()

print(infinite_values[infinite_values > 0])
print("\n--- NEGATIVE FLOW DURATION ---")

negative_duration = (df["Flow Duration"] < 0).sum()

print("Negative Flow Duration values:", negative_duration)
print("\n--- NEGATIVE FLOW DURATION ROWS ---")

negative_rows = df[df["Flow Duration"] < 0]

print(negative_rows[[
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Label"
]])
print("\n--- LABEL DISTRIBUTION ---")

print(df["Label"].value_counts())
print("\n--- OUTLIER CHECK: TOTAL FWD PACKETS ---")

Q1 = df["Total Fwd Packets"].quantile(0.25)
Q3 = df["Total Fwd Packets"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Total Fwd Packets"] < lower) |
    (df["Total Fwd Packets"] > upper)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)
print("Number of candidate outliers:", len(outliers))
