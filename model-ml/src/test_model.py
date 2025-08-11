import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ------------------------------
# 1. Initial settings
# ------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "no_show_model.pkl"      # model saved by train_model.py
TEST_DATA_PATH = ROOT_DIR / "data" / "test_split.csv"  # test data file
TARGET_COL = "No-show"                               # ground truth label column (0/1)

# ------------------------------
# 2. Load the model
# ------------------------------
print("[INFO] Loading saved model...")
model = joblib.load(str(MODEL_PATH))

# ------------------------------
# 3. Load test data
# ------------------------------
print("[INFO] Loading test data...")
df_test = pd.read_csv(TEST_DATA_PATH)

# Prepare features to match training
df_test['ScheduledDay'] = pd.to_datetime(df_test['ScheduledDay'])
df_test['AppointmentDay'] = pd.to_datetime(df_test['AppointmentDay'])
df_test['DaysBetween'] = (df_test['AppointmentDay'] - df_test['ScheduledDay']).dt.days.clip(lower=0)
df_test['AppointmentDayOfWeek'] = df_test['AppointmentDay'].dt.dayofweek
df_test['ScheduledHour'] = df_test['ScheduledDay'].dt.hour
df_test['IsWeekend'] = df_test['AppointmentDay'].dt.dayofweek.isin([5, 6]).astype(int)

numeric_features = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
    'Handcap', 'SMS_received', 'DaysBetween',
    'AppointmentDayOfWeek', 'ScheduledHour', 'IsWeekend'
]
categorical_features = ['Gender', 'Neighbourhood']

# Separate X and y with correct columns
X_test = df_test[numeric_features + categorical_features]
y_raw = df_test[TARGET_COL]
y_test = y_raw.map({'No': 0, 'Yes': 1}) if y_raw.dtype == 'O' else y_raw

# ------------------------------
# 4. Probability predictions
# ------------------------------
print("[INFO] Computing probabilities...")
y_probs = model.predict_proba(X_test)[:, 1]  # probabilitate pentru clasa 1 (no-show)

# ------------------------------
# 5. Threshold tuning
# ------------------------------
print("[INFO] Finding optimal threshold based on F1-score...")
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"[RESULT] Optimal threshold: {best_threshold:.2f} with F1-score: {f1_scores[best_idx]:.4f}")

# ------------------------------
# 6. Binary predictions with new threshold
# ------------------------------
y_pred_opt = (y_probs >= best_threshold).astype(int)

# ------------------------------
# 7. Final metrics
# ------------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_opt, digits=4))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_opt))

roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC: {roc_auc:.4f}")

# ------------------------------
# 8. Precision-Recall plot
# ------------------------------
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', label='Best threshold')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()
