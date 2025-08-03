# 🔍 Bank Customer Churn Prediction using ANN

This project predicts whether a customer will leave a bank (churn) based on various features using an Artificial Neural Network (ANN). The final model is deployed using **Streamlit** for an interactive web-based interface.

## 📌 Objective

To develop a deep learning model that can accurately classify bank customers as likely to churn or stay, helping financial institutions retain valuable clients.

---

## 🧠 Model Overview

The Artificial Neural Network was built using **TensorFlow** and **Keras**, trained on a processed dataset with binary classification as the objective.

### Features Used:
- Credit Score
- Geography (encoded)
- Gender (encoded)
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## 🔧 Tech Stack

- **Python** 🐍  
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn** – Data handling and visualization  
- **Sklearn** – Preprocessing, Label Encoding, Train-Test Split  
- **TensorFlow/Keras** – ANN Model building and training  
- **Streamlit** – Web app deployment  

---

## 📊 Model Architecture

- Input Layer with 11 features  
- Hidden Layers: Dense layers with ReLU activation  
- Dropout for regularization  
- Output Layer: Sigmoid activation for binary classification  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy

---

## 🚀 Streamlit App

The model is deployed via Streamlit allowing:
- Real-time predictions with user inputs
- Clean, interactive UI
- Lightweight, local deployment

To run the app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
