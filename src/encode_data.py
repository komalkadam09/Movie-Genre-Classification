import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv(
    "data/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID", "Genre", "Plot"]
)

# Encode Genre
encoder = LabelEncoder()
data["Genre"] = encoder.fit_transform(data["Genre"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=5000
)

X = vectorizer.fit_transform(data["Plot"])

print("TF-IDF Shape:")
print(X.shape)

print("\nEncoded Genres:")
print(data["Genre"].head())