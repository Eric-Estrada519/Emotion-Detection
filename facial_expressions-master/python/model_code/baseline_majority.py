import pandas as pd
from config import TRAIN_CSV, VAL_CSV, TEST_CSV

def compute_majority_baseline(csv_path):
    df = pd.read_csv(csv_path)
    majority_class = df["emotion"].value_counts().idxmax()
    majority_count = df["emotion"].value_counts().max()
    total = len(df)

    accuracy = majority_count / total
    return accuracy, majority_class

def main():
    train_acc, train_class = compute_majority_baseline(TRAIN_CSV)
    val_acc,   val_class   = compute_majority_baseline(VAL_CSV)
    test_acc,  test_class  = compute_majority_baseline(TEST_CSV)

    print("=== Majority Baseline Results ===")
    print(f"Train Majority Class: {train_class}, Accuracy: {train_acc:.4f}")
    print(f"Val Majority Class:   {val_class}, Accuracy: {val_acc:.4f}")
    print(f"Test Majority Class:  {test_class}, Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()



