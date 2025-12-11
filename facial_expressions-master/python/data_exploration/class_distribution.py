import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from config import LEGEND_CSV

#Code to load the csv
df = pd.read_csv(LEGEND_CSV)
df["emotion"] = df["emotion"].str.lower().str.strip()

#Code to count the number of classes
counts = df["emotion"].value_counts().reset_index()
counts.columns = ["Emotion", "Count"]
total = counts["Count"].sum()
counts["Percent"] = (counts["Count"] / total * 100).round(2)

#Printing the table
print("\n=== Class Distribution ===\n")
print(tabulate(counts, headers="keys", tablefmt="pretty"))

#A bar chart to help us visualize the data
plt.figure(figsize=(10, 6))
bars = plt.bar(counts["Emotion"], counts["Count"], color="purple")


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


plt.savefig("class_distribution_bar.png")
plt.show()
