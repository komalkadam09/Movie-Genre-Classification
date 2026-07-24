import joblib

# Load saved model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Take movie plot from user
plot = input("Enter Movie Plot: ")

# Convert plot to TF-IDF
plot_vector = vectorizer.transform([plot])

# Predict genre
prediction = model.predict(plot_vector)

# Convert number back to genre name
genre = encoder.inverse_transform(prediction)

print("\nPredicted Genre:")
print(genre[0])