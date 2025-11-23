import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from config import LEGEND_CSV

# Load CSV
df = pd.read_csv(LEGEND_CSV)
df["emotion"] = df["emotion"].str.lower().str.strip()

# Count classes
counts = df["emotion"].value_counts().reset_index()
counts.columns = ["Emotion", "Count"]
total = counts["Count"].sum()
counts["Percent"] = (counts["Count"] / total * 100).round(2)

# Print table
print("\n=== Class Distribution ===\n")
print(tabulate(counts, headers="keys", tablefmt="pretty"))

# -------- BAR CHART ADDED --------
plt.figure(figsize=(10, 6))
bars = plt.bar(counts["Emotion"], counts["Count"], color="purple")

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 50,
             str(height),
             ha="center", va="bottom", fontsize=9)

plt.title("Class Distribution of Emotion Dataset (Counts)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

# Save & show
plt.savefig("class_distribution_bar.png")
plt.show()
