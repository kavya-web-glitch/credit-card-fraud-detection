# 💳 Credit Card Fraud Detection

## 📌 Project Overview

This project uses Machine Learning to detect fraudulent credit card transactions.

A Random Forest model is trained with SMOTE to handle the imbalanced fraud dataset.

## 🎯 Objectives

- Detect fraudulent transactions
- Reduce missed fraud cases
- Calculate fraud probability
- Classify transactions into risk levels
- Provide real-time fraud monitoring

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SMOTE
- Streamlit
- Matplotlib
- Seaborn

## 🤖 Machine Learning Model

**Random Forest + SMOTE**

The model uses transaction information such as:

- Transaction amount
- Transaction hour
- Foreign transaction
- Location mismatch
- Device trust score
- Transaction velocity
- Cardholder age
- Merchant category

## 📊 Model Performance

- ROC-AUC: 0.99
- Fraud Recall: 73.33%

## 🖥️ Application

The project contains:

- Fraud prediction application
- Analytics dashboard
- Transaction history
- Fraud alerts
- Risk-level classification
- Model performance analysis

## 📁 Project Structure

```text
Fraud_detection_Project/
│
├── app/
├── dashboard/
├── data/
├── models/
├── notebooks/
├── src/
└── README.md

🚀 How to Run

Run the prediction application:
python -m streamlit run app/app.py

Run the analytics dashboard:
python -m streamlit run dashboard/dashboard.py

## 👩‍💻 Project Purpose

The purpose of this project is to develop an end-to-end Credit Card Fraud Detection system using Data Science and Machine Learning.

The system analyzes transaction details and identifies potentially fraudulent transactions using a Random Forest model with SMOTE to handle imbalanced data.

It also provides:

- Real-time fraud prediction
- Fraud probability calculation
- Low, Medium, and High Risk classification
- Transaction history tracking
- Fraud alerts for high-risk transactions
- Interactive analytics dashboard
- Model performance evaluation
- Confusion Matrix and ROC Curve analysis
- Feature importance analysis
- Transaction report download

The project demonstrates the complete Machine Learning workflow from data preprocessing and model training to evaluation, deployment, and real-time monitoring using Streamlit.