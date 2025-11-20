import pandas as pd
from tabulate import tabulate
from config import LEGEND_CSV

df = pd.read_csv(LEGEND_CSV)
df["emotion"] = df["emotion"].str.lower()

counts = df["emotion"].value_counts().reset_index()
counts.columns = ["Emotion", "Count"]
total = counts["Count"].sum()
counts["Percent"] = (counts["Count"] / total * 100).round(2)

print("\n=== Class Distribution ===\n")
print(tabulate(counts, headers="keys", tablefmt="pretty"))



# We can talk about how undistributed the data is.