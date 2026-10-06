# Movie Genre Classification using Machine Learning

## Project Description
This project predicts the genre of a movie based on its plot summary using Machine Learning. It uses TF-IDF for feature extraction and Logistic Regression for classification.

## Dataset
IMDb Genre Classification Dataset

## Algorithm Used
- Logistic Regression

## Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency)

## Libraries Used
- pandas
- scikit-learn
- joblib

## Project Structure

MovieGenreClassification/
│
├── data/
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── encode_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── predict.py
│
├── main.py
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train_model.py
```

3. Run the prediction program:

```bash
python main.py
```

## Model Performance

- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF
- Accuracy: 57.91%

## Author
Komal Kadam
