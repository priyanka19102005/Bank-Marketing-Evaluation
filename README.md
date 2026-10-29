# 🏦 Bank Marketing Evaluation & Term Deposit Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=priyanka19102005/Bank-Marketing-Evaluation&branch=main&mainModule=app.py)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application deployed with Streamlit that analyzes marketing campaign efficacy and predicts whether bank clients will subscribe to a term deposit.

---

## 📌 Project Overview

This project focuses on predicting customer response to direct marketing campaigns (phone calls) for bank term deposits. It includes a complete workflow:
- **Exploratory Data Analysis (EDA)** & visualization of customer demographics and financial factors.
- **Feature Engineering & Preprocessing**: Median imputation, Standard Scaling, and One-Hot Encoding wrapped in Scikit-Learn Pipelines.
- **Machine Learning Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Tuned Decision Tree (optimized depth & minimum split constraints to prevent overfitting)
- **Evaluation & Diagnostics**: Accuracy, Precision, Recall, F1 Score, ROC Curves (AUC), and Confusion Matrices.
- **Interactive Web App**: Single customer prediction simulator, model evaluation dashboard, dataset explorer, and batch CSV predictions.

---

## 🎯 Target Variable

- **`0` (No)**: Customer did not subscribe to a term deposit.
- **`1` (Yes)**: Customer subscribed to a term deposit.

---

## 📊 Dataset Features

The dataset (`bank_marketing.csv`) contains:
- **Demographics**: `age`, `job`, `marital`, `education`
- **Credit & Banking**: `default`, `balance`, `housing`, `loan`
- **Current Campaign**: `contact`, `day`, `month`, `duration`, `campaign`
- **Previous Campaigns**: `pdays`, `previous`, `poutcome`
- **Target**: `y`

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/priyanka19102005/Bank-Marketing-Evaluation.git
   cd Bank-Marketing-Evaluation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push your latest code to GitHub (`main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and click **New App**.
4. Set:
   - **Repository**: `priyanka19102005/Bank-Marketing-Evaluation`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**
