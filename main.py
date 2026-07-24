import joblib

print("========================================")
print("      MOVIE GENRE CLASSIFICATION")
print("========================================")

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# User input
plot = input("\nEnter Movie Plot:\n")

# Convert to TF-IDF
plot_vector = vectorizer.transform([plot])

# Predict genre
prediction = model.predict(plot_vector)

# Convert back to original genre
genre = encoder.inverse_transform(prediction)

print("\n==============================")
print("Predicted Movie Genre:")
print(genre[0])