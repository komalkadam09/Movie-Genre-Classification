import pandas as pd

# Load dataset
data = pd.read_csv(
    "data/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "Genre", "Plot"]
)

print("Original Shape:")
print(data.shape)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

print("\nShape After Removing Missing Values:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())