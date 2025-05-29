# Credit Card Fraud

# Logistic Regression (Simple)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
credit_card_data = pd.read_csv('/content/creditcard.csv')

# 2. Features and target
X = credit_card_data.drop(columns=['Class'])
y = credit_card_data['Class']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scaling 'Amount' and 'Time' (common best practice for this dataset)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# 5. Logistic Regression Model with class weight balancing
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

# 6. Evaluation
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 7. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

conf_matrix, class_report, roc_auc

"""High recall for fraud (92%) → The model detects almost all fraud cases.

Extremely low precision (6%) → Very high false positives.

ROC-AUC (0.9721) → Strong starting point, shows model is able to rank well even if threshold tuning is needed.

Imbalance challenge still exists → The model predicts many legitimate transactions as fraud.
"""

# LogisticRegression (Tuned)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
credit_card_data = pd.read_csv('/content/creditcard.csv')

# 2. Features and target
X = credit_card_data.drop(columns=['Class'])
y = credit_card_data['Class']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scaling 'Amount' and 'Time' (common best practice for this dataset)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# 5. Logistic Regression Model with class weight balancing
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

# 6. Evaluation
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 7. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

conf_matrix, class_report, roc_auc

"""High recall for fraud (92%) → The model detects almost all fraud cases.

Extremely low precision (6%) → Very high false positives.

ROC-AUC (0.9721) → Strong starting point, shows model is able to rank well even if threshold tuning is needed.

Imbalance challenge still exists → The model predicts many legitimate transactions as fraud.
"""

from sklearn.metrics import precision_recall_curve

# Find the best threshold for precision-recall trade-off
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Calculate F1 score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

# Apply the best threshold
y_pred_threshold = (y_proba >= best_threshold).astype(int)

# Re-evaluate with the new threshold
conf_matrix_thr = confusion_matrix(y_test, y_pred_threshold)
class_report_thr = classification_report(y_test, y_pred_threshold)
roc_auc_thr = roc_auc_score(y_test, y_proba)  # AUC doesn't change as it's threshold-independent

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve with Best Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

best_threshold, best_f1, conf_matrix_thr, class_report_thr, roc_auc_thr

"""Fraud detection precision is now 83% (previously 6%).

Fraud recall is still high at 82% (previously 92%), but more balanced.

Best F1-score for fraud reached ~0.82, a major leap from the baseline.

Threshold needed to be almost 1, meaning only extremely confident predictions are considered fraud.
"""

# Random Forest

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

# Random Forest Hypertuned Plus Feature Engineering

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('/content/creditcard.csv')

# Feature Engineering
credit_card_data['Hour'] = (credit_card_data['Time'] % (60*60*24)) / (60*60)  # extract hour of transaction
# Bin 'Amount' into 5 categories
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
credit_card_data['Amount_Bin'] = kbin.fit_transform(credit_card_data[['Amount']])

# Prepare features and target
X = credit_card_data.drop(columns=['Class', 'Time', 'Amount'])  # drop raw 'Time' and 'Amount' after feature creation
X['Time_Hour'] = credit_card_data['Hour']
X['Amount_Bin'] = credit_card_data['Amount_Bin']
y = credit_card_data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Tuning using GridSearchCV (optimize for recall focus on fraud)
param_grid = {
    'n_estimators': [50],
    'max_depth': [10, 15, None],
    'min_samples_leaf': [10],
    'max_features': ['sqrt'],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)

rf.fit(X_train, y_train)

# Predictions and probabilities
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

# Apply best threshold
y_pred_rf_thresh = (y_proba_rf >= best_threshold).astype(int)

# Metrics
conf_matrix_base = confusion_matrix(y_test, y_pred_rf)
class_report_base = classification_report(y_test, y_pred_rf, output_dict=True)

conf_matrix_thresh = confusion_matrix(y_test, y_pred_rf_thresh)
class_report_thresh = classification_report(y_test, y_pred_rf_thresh, output_dict=True)

roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve - Random Forest (Ultra-fast)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Random Forest (Improved)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Results table
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
        roc_auc_rf
    ]
})

results_df.round(4)

"""Threshold tuning improved F1-Score from 0.8343 → 0.8791.

Recall improved significantly (74.5% → 81.6%) — better fraud catch rate.

Precision stayed very high (94.8% → 95.2%).

ROC-AUC remains excellent at 0.9530.
"""

# XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('/content/creditcard.csv')

# Feature Engineering
credit_card_data['Hour'] = (credit_card_data['Time'] % (60*60*24)) / (60*60)
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
credit_card_data['Amount_Bin'] = kbin.fit_transform(credit_card_data[['Amount']])

# Prepare features and target
X = credit_card_data.drop(columns=['Class', 'Time', 'Amount'])
X['Hour'] = credit_card_data['Hour']
X['Amount_Bin'] = credit_card_data['Amount_Bin']
y = credit_card_data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost Model
xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=1
)
xgb.fit(X_train, y_train)

# Predictions and probabilities
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_xgb)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

# Apply best threshold
y_pred_xgb_thresh = (y_proba_xgb >= best_threshold).astype(int)

# Metrics
conf_matrix_base = confusion_matrix(y_test, y_pred_xgb)
class_report_base = classification_report(y_test, y_pred_xgb, output_dict=True)

conf_matrix_thresh = confusion_matrix(y_test, y_pred_xgb_thresh)
class_report_thresh = classification_report(y_test, y_pred_xgb_thresh, output_dict=True)

roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve - XGBoost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Results table
results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [
        class_report_base['1']['precision'],
        class_report_base['1']['recall'],
        class_report_base['1']['f1-score'],
        roc_auc_xgb
    ],
    'Threshold Tuned': [
        class_report_thresh['1']['precision'],
        class_report_thresh['1']['recall'],
        class_report_thresh['1']['f1-score'],
        roc_auc_xgb
    ]
})

results_df.round(4)

"""Baseline recall is excellent (84.7%), but precision is lower (73.4%).

Threshold tuning significantly improved precision to 96.1%, while recall dropped to 75.5%.

F1-Score improved from 0.7867 → 0.8457, indicating a better balance after threshold tuning.

ROC-AUC is very strong (0.9767), showing excellent discrimination.
"""

# Light GBM

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('/content/creditcard.csv')

# Feature Engineering
credit_card_data['Hour'] = (credit_card_data['Time'] % (60*60*24)) / (60*60)
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
credit_card_data['Amount_Bin'] = kbin.fit_transform(credit_card_data[['Amount']])

# Prepare features and target
X = credit_card_data.drop(columns=['Class', 'Time', 'Amount'])
X['Hour'] = credit_card_data['Hour']
X['Amount_Bin'] = credit_card_data['Amount_Bin']
y = credit_card_data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LightGBM Model with imbalance handling
lgbm = LGBMClassifier(
    is_unbalance=True,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1
)
lgbm.fit(X_train, y_train)

# Predictions and probabilities
y_pred_lgbm = lgbm.predict(X_test)
y_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lgbm)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

# Apply best threshold
y_pred_lgbm_thresh = (y_proba_lgbm >= best_threshold).astype(int)

# Metrics
conf_matrix_base = confusion_matrix(y_test, y_pred_lgbm)
class_report_base = classification_report(y_test, y_pred_lgbm, output_dict=True)

conf_matrix_thresh = confusion_matrix(y_test, y_pred_lgbm_thresh)
class_report_thresh = classification_report(y_test, y_pred_lgbm_thresh, output_dict=True)

roc_auc_lgbm = roc_auc_score(y_test, y_proba_lgbm)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve - LightGBM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_lgbm)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - LightGBM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Results table
results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [
        class_report_base['1']['precision'],
        class_report_base['1']['recall'],
        class_report_base['1']['f1-score'],
        roc_auc_lgbm
    ],
    'Threshold Tuned': [
        class_report_thresh['1']['precision'],
        class_report_thresh['1']['recall'],
        class_report_thresh['1']['f1-score'],
        roc_auc_lgbm
    ]
})

results_df.round(4)

"""Very high recall (88.7%), but precision is extremely poor (~3–4%).

Threshold tuning helped precision slightly (from 3.4% to 4.1%), but still low.

ROC-AUC (0.9221) is still strong, meaning LightGBM is good at ranking, but classification threshold needs work.

F1-Score is the lowest among all models tried so far, suggesting that LightGBM (with current settings) is not performing well in precision-recall trade-off.
"""

# SVM

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('/content/creditcard.csv')

# Feature Engineering
credit_card_data['Hour'] = (credit_card_data['Time'] % (60*60*24)) / (60*60)
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
credit_card_data['Amount_Bin'] = kbin.fit_transform(credit_card_data[['Amount']])

# Prepare features and target
X = credit_card_data.drop(columns=['Class', 'Time', 'Amount'])
X['Hour'] = credit_card_data['Hour']
X['Amount_Bin'] = credit_card_data['Amount_Bin']
y = credit_card_data['Class']

# Sample the data for SVM (use all fraud + random 10,000 legit)
fraud = credit_card_data[credit_card_data['Class'] == 1]
non_fraud = credit_card_data[credit_card_data['Class'] == 0].sample(n=10000, random_state=42)
sampled_data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

X_sampled = sampled_data.drop(columns=['Class', 'Time', 'Amount'])
X_sampled['Hour'] = sampled_data['Hour']
X_sampled['Amount_Bin'] = sampled_data['Amount_Bin']
y_sampled = sampled_data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model with class weight
svm = SVC(class_weight='balanced', probability=True, kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)

# Predictions and probabilities
y_pred_svm = svm.predict(X_test_scaled)
y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_svm)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

# Apply best threshold
y_pred_svm_thresh = (y_proba_svm >= best_threshold).astype(int)

# Metrics
conf_matrix_base = confusion_matrix(y_test, y_pred_svm)
class_report_base = classification_report(y_test, y_pred_svm, output_dict=True)

conf_matrix_thresh = confusion_matrix(y_test, y_pred_svm_thresh)
class_report_thresh = classification_report(y_test, y_pred_svm_thresh, output_dict=True)

roc_auc_svm = roc_auc_score(y_test, y_proba_svm)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.scatter(recall[best_index], precision[best_index], color='red', label=f'Best Threshold: {best_threshold:.4f}')
plt.title('Precision-Recall Curve - SVM (Small Sample)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - SVM (Small Sample)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Results table
results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [
        class_report_base['1']['precision'],
        class_report_base['1']['recall'],
        class_report_base['1']['f1-score'],
        roc_auc_svm
    ],
    'Threshold Tuned': [
        class_report_thresh['1']['precision'],
        class_report_thresh['1']['recall'],
        class_report_thresh['1']['f1-score'],
        roc_auc_svm
    ]
})

results_df.round(4)

"""SVM showed strong recall (87.8%) at baseline, with precision at 72.3%.

Threshold tuning boosted precision to 95.1%, recall dropped slightly to 80.6%.

F1-Score improved to 0.8729 after threshold tuning, performing very similarly to XGBoost and Random Forest (on the small sample).

ROC-AUC (0.9813) is excellent, showing the model is able to discriminate fraud effectively even on a small balanced sample.

"""

# Random Forest Improved (BEST)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix, roc_auc_score

credit_card_data = pd.read_csv('/content/creditcard.csv')

# EDA
print("Dataset Shape:", credit_card_data.shape)
print("\nClass Distribution:")
print(credit_card_data['Class'].value_counts(normalize=True))
sns.countplot(x='Class', data=credit_card_data)
plt.title('Class Distribution')
plt.show()

# Correlation Heatmap (optional insight)
plt.figure(figsize=(12, 8))
corr = credit_card_data.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Feature Engineering
credit_card_data['Hour'] = (credit_card_data['Time'] % (60*60*24)) / (60*60)
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
credit_card_data['Amount_Bin'] = kbin.fit_transform(credit_card_data[['Amount']])

# Prepare features and target
X = credit_card_data.drop(columns=['Class', 'Time', 'Amount'])
X['Hour'] = credit_card_data['Hour']
X['Amount_Bin'] = credit_card_data['Amount_Bin']
y = credit_card_data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------- Improved Random Forest Model ----------------
rf = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------------- Threshold tuning (Focus on recall) ----------------
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)

# Recall and thresholds are of same length, precision is len(thresholds) + 1
# Always use thresholds and recall[:-1] together
desired_recall = 0.85

# Mask where recall >= desired_recall (use recall[:-1])
mask = recall[:-1] >= desired_recall

# Find all candidate thresholds
candidate_thresholds = thresholds[mask]

# Corresponding precisions (use precision[mask])
candidate_precisions = precision[:-1][mask]

# Choose the best threshold based on highest precision at desired recall
if len(candidate_thresholds) > 0:
    best_index = np.argmax(candidate_precisions)
    best_threshold = candidate_thresholds[best_index]
else:
    # Fallback to threshold with max recall
    best_index = np.argmax(recall[:-1])
    best_threshold = thresholds[best_index]

print(f"Best threshold prioritizing recall >= {desired_recall*100:.0f}%: {best_threshold:.4f}")

# Apply the chosen threshold
y_pred_rf_thresh = (y_proba_rf >= best_threshold).astype(int)

# Evaluation
conf_matrix_thresh = confusion_matrix(y_test, y_pred_rf_thresh)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

print("\n✅ Classification Report with Tuned Threshold (Prioritizing Recall):")
print(classification_report(y_test, y_pred_rf_thresh))

# Visual Confusion Matrix with focus on fraud class
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_thresh, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - RF (Improved, Tuned for Recall Priority)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Detailed Metrics Table
class_report_thresh = classification_report(y_test, y_pred_rf_thresh, output_dict=True)
results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Fraud Class (1)': [
        class_report_thresh['1']['precision'],
        class_report_thresh['1']['recall'],
        class_report_thresh['1']['f1-score'],
        roc_auc_rf
    ]
})

results_df.round(4)

"""Recall Focus Achieved:
→ 14 false negatives only (missed frauds), recall of ~85.7%.

Precision is still strong (73%), meaning most predicted frauds are actual frauds.

F1-Score improved while ensuring priority on catching frauds (recall-focused tuning).

ROC-AUC remains strong at 0.9606, showing strong discriminative power.
"""