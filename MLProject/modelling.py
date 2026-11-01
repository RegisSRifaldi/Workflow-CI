
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# 1. Load dataset hasil preprocessing
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "bank_marketing_preprocessing")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print("Dataset berhasil dimuat.")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ==========================================
# 2. Training model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==========================================
# 3. Evaluasi model
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

# ==========================================
# 4. Logging parameter dan metrics
# ==========================================

mlflow.log_param("model_type", "RandomForestClassifier")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_param("random_state", 42)
mlflow.log_param("class_weight", "balanced")

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)

# ==========================================
# 5. Simpan confusion matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_No", "Actual_Yes"],
    columns=["Predicted_No", "Predicted_Yes"]
)

cm_path = os.path.join(BASE_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_path)

mlflow.log_artifact(cm_path)

# ==========================================
# 6. Simpan model
# ==========================================

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="bank_marketing_model"
)

# ==========================================
# 7. Tampilkan hasil evaluasi
# ==========================================

print("\n=== HASIL EVALUASI MODEL ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nTraining dan logging MLflow berhasil.")
