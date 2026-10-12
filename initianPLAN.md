# Phishing Website URL Detection Using Machine Learning

## 1. Project Overview

Build a simple machine learning application that classifies website URLs as **legitimate or phishing** using a Random Forest Classifier. The project will include model training, evaluation, visualizations, and a basic Streamlit interface.

## 2. Objectives

- Load a phishing URL dataset from Hugging Face.
- Perform basic data exploration and preprocessing.
- Train a Random Forest classification model.
- Evaluate the model using suitable performance metrics.
- Visualize the results using graphs.
- Create a simple Streamlit interface for predictions.

## 3. Dataset

**Source:** [pirocheto/phishing-url — Hugging Face](https://huggingface.co/datasets/pirocheto/phishing-url)

Load the training and testing data from the provided Parquet files:

- `data/train.parquet`
- `data/test.parquet`

Inspect the columns, target labels, data types, and missing values before preprocessing.

## 4. Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Hugging Face Datasets / Parquet support

## 5. Implementation Plan

### Step 1: Data Loading and Exploration

- Load the training and testing datasets.
- Display sample records, dataset shape, and column names.
- Identify the target column and input features.
- Check missing values, duplicates, and class distribution.

### Step 2: Data Preprocessing

- Handle missing values where necessary.
- Remove unsuitable columns, such as identifiers, if present.
- Separate input features and target labels.
- Encode categorical variables if required.
- Ensure training and testing features are consistent.

### Step 3: Model Training

- Use **Random Forest Classifier** as the primary algorithm.
- Train the model using the training dataset.
- Save the trained model and any required preprocessing objects.

### Step 4: Model Evaluation

Calculate the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

Generate a confusion matrix to examine correct and incorrect predictions.

### Step 5: Data Visualization

Create the following plots:

- Class distribution chart.
- Confusion matrix heatmap.
- Feature-importance chart.

### Step 6: Streamlit Interface

Develop a basic interface that allows users to:

- Enter or select URL feature values supported by the trained model.
- Click a Predict button.
- View the predicted class: Legitimate or Phishing.

The interface must use the same features and preprocessing as the trained model. Do not fabricate missing features or present unsupported predictions.

### Step 7: Documentation

Prepare a `README.md` file containing:

- Project description and objectives.
- Dataset source.
- Installation instructions.
- Commands to train and run the application.
- Evaluation metrics and results.
- Project limitations.

## 6. Project Structure

```text
phishing-url-detection/
│
├── data/
│   ├── train.parquet
│   └── test.parquet
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── phishing_model.pkl
└── outputs/
    ├── class_distribution.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

The Parquet files can be loaded directly from Hugging Face, so downloading them into the project directory is optional.

## 7. Expected Deliverables

1. Python source code.
2. Trained machine learning model.
3. Evaluation metrics and graphs.
4. Streamlit prediction interface.
5. README documentation.
6. Assignment report in PDF format.

## 8. Scope and Constraints

- Use only one main ML algorithm to keep implementation simple.
- Avoid deep learning and unnecessary model comparisons.
- Reuse the dataset's official train/test split where appropriate.
- Keep the interface minimal.
- Report actual evaluation results; do not assume a target accuracy in advance.
- Clearly document limitations, especially if the dataset's features cannot be calculated from a raw URL alone.

## 9. Completion Goal

Complete a functional, reproducible ML assignment with a trained Random Forest model, evaluation visualizations, a basic Streamlit interface, and clear documentation. Prioritize correctness and simplicity over additional features.