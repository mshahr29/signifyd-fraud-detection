# E-commerce Fraud Detection with Multi-Stage Models | Inspired by Signifyd

This repository contains a two-stage machine learning pipeline inspired by **Signifyd**, a global leader in e-commerce fraud prevention. The models are designed to help e-commerce platforms **detect fraudulent transactions**, **reduce chargebacks**, and **approve more legitimate orders**, mirroring the real-world workflow of digital fraud defense.

---

## 📌 Project Overview

E-commerce platforms are constantly targeted by fraudsters exploiting stolen credentials, compromised devices, and loopholes at checkout. To combat this, I built a **multi-stage fraud detection system** using real-world-inspired features.

🔍 The two models reflect **distinct stages of the order funnel**:

1. **Transaction Stage (Pre-checkout)**

   * Detects fraudulent behavior based on device ID, IP location, signup timestamp, and browsing patterns.
   * Predicts if a transaction attempt is suspicious before a payment is initiated.

2. **Checkout Stage (Post-payment attempt)**

   * Evaluates payment fraud likelihood using transaction amount, billing info, card metadata, and velocity features.
   * Aims to flag fraud while minimizing false positives for legitimate buyers.

---

## 🧠 Models Used

### Stage 1: Transaction Model

* Logistic Regression with SMOTE
* SMOTE & class weighting to handle severe class imbalance
* Features: `device_id`, `ip_address`, `signup_time`, `purchase_time`, `user_history`, etc.

### Stage 2: Checkout Model

* Random Forest with time-based features and risk profiling
* Feature engineering includes `hour`, `amount_bin`.

---

## 🧪 Techniques & Tools

* Python (pandas, scikit-learn, logistic regression, xgboost, lightgbm)
* Imbalanced learning (SMOTE, ADASYN, class\_weight)
* Feature engineering for time-series and device fingerprinting
* Model evaluation using:

  * Recall & Precision
  * AUC-ROC
  * F1
  * Confusion Matrix with business-focused metrics (false positives vs false negatives)

---

## 📊 Results

* **Transaction Model Recall:** *72%*
* **Checkout Model Recall:** *86%*
* **ROC-AUC (Both models):** *80%+*
* Business impact modeled as reduced chargebacks and higher approval rates

---

## 🔧 Folder Structure

```
📁 Ecommerce-Fraud-Detection
│
├── data/                # Sample or synthetic fraud datasets
├── models/              # Python scripts for the two models
│   ├── transaction_model.py
│   └── checkout_model.py
└── README.md            # Project documentation
```

---

## 🌐 Real-World Relevance

This project draws on Signifyd’s multi-layered approach to fraud detection. By simulating both the **pre-purchase** and **post-payment** checkpoints, it reflects the layered security architecture used by major e-commerce platforms to:

* Increase order approval rates
* Reduce revenue lost to false declines
* Cut down chargebacks and fraud losses

---

## 📌 Future Work

* Integrate user behavioral modeling (e.g., browsing session sequences)
* Build a real-time scoring API with FastAPI or Flask
* Add graph-based fraud detection (e.g., shared device or IP clusters)

---
