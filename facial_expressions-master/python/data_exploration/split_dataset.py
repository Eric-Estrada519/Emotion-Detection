import pandas as pd
from sklearn.model_selection import train_test_split
from config import LEGEND_CSV
import os

df = pd.read_csv(LEGEND_CSV)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["emotion"],     # keep balanced classes
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["emotion"],
    random_state=42
)

# Save
data_dir = os.path.dirname(LEGEND_CSV)
train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

print("Done!")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
