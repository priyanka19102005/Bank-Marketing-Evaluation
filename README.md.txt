
# Bank Marketing Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a bank customer will subscribe to a term deposit based on customer information and marketing campaign data.

The project follows a complete machine learning workflow, starting from data loading and cleaning to exploratory data analysis, feature engineering, WOE, IV, VIF analysis, model building, hyperparameter tuning, and model evaluation.

Two main machine learning algorithms are implemented:

- Logistic Regression
- Decision Tree

Hyperparameter tuning is also performed on the Decision Tree model to improve its performance.

---

## 🎯 Project Objective

The main objective of this project is to predict whether a customer will subscribe to a bank term deposit.

### Target Variable

- `0` → Customer did not subscribe
- `1` → Customer subscribed

The project also compares different machine learning models using multiple evaluation metrics and identifies the best-performing model.

---

## 📊 Dataset

The project uses a Bank Marketing dataset containing customer and marketing campaign information.

The dataset includes features related to:

- Age
- Job
- Marital Status
- Education
- Account Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign Information
- Previous Contacts
- Previous Campaign Outcome

### Target

The target variable represents whether the customer subscribed to a term deposit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Spyder / Jupyter Notebook
- Git
- GitHub

---

## 🔄 Project Workflow

```text
Data Loading
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Engineering
      ↓
Weight of Evidence (WOE)
      ↓
Information Value (IV)
      ↓
Variance Inflation Factor (VIF)
      ↓
Train-Test Split
      ↓
Logistic Regression
      ↓
Decision Tree
      ↓
Hyperparameter Tuning
      ↓
Tuned Decision Tree
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Best Model