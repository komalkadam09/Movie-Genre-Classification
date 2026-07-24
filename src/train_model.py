import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("==============================")
print("Model Accuracy:")
print(accuracy)

print("\n==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel saved successfully.")