# Wine Type Classification Using Machine Learning

## 1. Project Overview

This project develops a machine learning classification system to identify whether a wine sample is Red Wine or White Wine based on its physicochemical properties.

The system uses five supervised machine learning algorithms and compares their classification performance. A Streamlit web application allows users to upload wine data, select a trained model, generate predictions, and evaluate model performance.

## 2. Dataset Description

The dataset is obtained from the UCI Machine Learning Repository – Wine Quality Dataset.

- Red wine samples: 1,599
- White wine samples: 4,898
- Total samples: 6,497
- Input features: 12
- Target variable: `Wine_Type`

Target encoding:
- `0` – Red Wine
- `1` – White Wine

### Input Features

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality

## 3. Data Preprocessing

- Loaded red and white wine datasets.
- Added a binary target variable called `Wine_Type`.
- Combined both datasets.
- Randomly shuffled the combined dataset.
- Checked for missing values.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve class proportions.

## 4. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

StandardScaler was used in the Logistic Regression, K-Nearest Neighbors, and Support Vector Machine pipelines.

## 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Confusion Matrix

### Model Comparison

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.9946 | 0.9959 |
| Decision Tree | 0.9892 | 0.9897 |
| K-Nearest Neighbors | 0.9946 | 0.9983 |
| Gaussian Naive Bayes | 0.9754 | 0.9881 |
| Support Vector Machine | 0.9977 | 0.9977 |

The Support Vector Machine achieved the highest test accuracy among the five evaluated models.

## 6. Streamlit Web Application

The project includes an interactive Streamlit application.

Application features:

- Select one of the five trained models.
- Upload a CSV dataset.
- Preview uploaded data.
- Predict Red Wine or White Wine.
- Display evaluation metrics when actual target labels are provided.
- Display a confusion matrix.
- Compare the performance of all five models.
- Download prediction results as a CSV file.

## 7. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 8. Project Structure

```text
ML_Classification_Assignment/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── wine_quality_combined.csv
├── winequality-red.csv
├── winequality-white.csv
├── winequality.names
├── wine_quality.zip
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k-nearest_neighbors.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── confusion matrix CSV files
```

## 9. How to Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and save the models:

```bash
python train_models.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## 10. Conclusion

This project demonstrates the application of supervised machine learning algorithms for binary wine-type classification.

Five classification algorithms were trained and compared. The Support Vector Machine achieved the highest accuracy of approximately 99.77% on the held-out test dataset.

The Streamlit application provides an interactive interface for testing the trained models and generating predictions.

## 11. Dataset Reference

UCI Machine Learning Repository:

Wine Quality Dataset  
https://archive.ics.uci.edu/dataset/186/wine+quality
