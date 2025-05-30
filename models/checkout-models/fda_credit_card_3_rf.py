# -*- coding: utf-8 -*-
"""FDA_Credit_Card_3_rf.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z8eNfIP7jwou4p7y50fif_d9Gt9tJUub
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('/content/creditcard.csv')

# Feature selection and preprocessing
X = credit_card_data.drop(columns=['Class'])
y = credit_card_data['Class']

# Train-Test Split (stratified due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling 'Amount' and 'Time'
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Random Forest Model (baseline) with adjusted n_jobs
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)

# Predictions and probabilities
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Precision-Recall Curve and Threshold Tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

# Apply best threshold
y_pred_rf_thresh = (y_proba_rf >= best_threshold).astype(int)

# Metrics evaluation
conf_matrix_base = confusion_matrix(y_test, y_pred_rf)
class_report_base = classification_report(y_test, y_pred_rf, output_dict=True)

conf_matrix_thresh = confusion_matrix(y_test, y_pred_rf_thresh)
class_report_thresh = classification_report(y_test, y_pred_rf_thresh, output_dict=True)

roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve - Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [
        class_report_base['1']['precision'],
        class_report_base['1']['recall'],
        class_report_base['1']['f1-score'],
        roc_auc_rf
    ],
    'Threshold Tuned': [
        class_report_thresh['1']['precision'],
        class_report_thresh['1']['recall'],
        class_report_thresh['1']['f1-score'],
        roc_auc_rf  # AUC remains the same regardless of threshold
    ]
})

results_df.round(4)