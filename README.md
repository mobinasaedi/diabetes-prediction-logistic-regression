# Diabetes Prediction Using Logistic Regression

## Project Overview

This project applies Logistic Regression, a supervised machine learning algorithm, to a binary classification task using a diabetes dataset.

The objective is to build a classification model that analyzes patient-related numerical features and predicts the corresponding outcome recorded in the dataset.

## Objectives

- Understand and explore the dataset.
- Perform data preprocessing and handle missing values.
- Separate input features and target variables.
- Split the dataset into training and testing sets.
- Apply feature scaling.
- Build and train a Logistic Regression model.
- Evaluate model performance using classification metrics.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn

## Machine Learning Workflow

1. **Data Loading:** Import the diabetes dataset using Pandas.
2. **Exploratory Data Analysis:** Examine dataset structure and statistical information.
3. **Data Preprocessing:** Identify invalid zero values and handle missing data using median imputation.
4. **Feature Selection:** Separate independent variables (X) from the target variable (y).
5. **Data Splitting:** Divide the dataset into training (80%) and testing (20%) subsets.
6. **Feature Scaling:** Apply StandardScaler to standardize numerical features.
7. **Model Training:** Train a Logistic Regression classifier.
8. **Model Evaluation:** Evaluate predictions using Accuracy, Confusion Matrix, Precision, Recall, F1-score, and ROC-AUC.

## Project Structure

```text
diabetes-prediction-logistic-regression/
│
├── diabetes.csv
├── diabetes_prediction.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/diabetes-prediction-logistic-regression.git
```

Navigate to the project directory:

```bash
cd diabetes-prediction-logistic-regression
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Make sure the dataset file (`diabetes.csv`) is located in the project directory.

Run the Python script:

```bash
python diabetes_prediction.py
```

The script trains the Logistic Regression model and displays its evaluation results.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC

## Limitations

This project is an educational machine learning exercise. Its predictions are based on the provided dataset and should not be interpreted as medical diagnoses or used for clinical decision-making.

The model's performance depends on the dataset's quality, characteristics, and representativeness.

## Future Improvements

- Perform feature analysis and visualization.
- Compare Logistic Regression with other classification algorithms.
- Apply hyperparameter tuning.
- Evaluate model performance using cross-validation.
- Explore additional classification metrics.

## Author

**Mobina Saedi**

Computer Engineering Student | Machine Learning Enthusiast
