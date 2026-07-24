import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=5000
)

X = vectorizer.fit_transform(data["Plot"])

# Target
y = data["Genre"]

print("Features Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)