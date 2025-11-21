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

"""
+---+-----------+-------+---------+
|   |  Emotion  | Count | Percent |
+---+-----------+-------+---------+
| 0 |  neutral  | 6778  |  51.63  |
| 1 | happiness | 5453  |  41.54  |
| 2 | surprise  |  362  |  2.76   |
| 3 |   anger   |  244  |  1.86   |
| 4 |  sadness  |  195  |  1.49   |
| 5 |  disgust  |  71   |  0.54   |
| 6 |   fear    |  16   |  0.12   |
| 7 | contempt  |   9   |  0.07   |
+---+-----------+-------+---------+

"""