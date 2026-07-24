import pandas as pd

# Load the training dataset
data = pd.read_csv(
    "data/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "Genre", "Plot"]
)

print("========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== COLUMN NAMES ==========")
print(data.columns)