import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('Churn_Modelling.csv')


X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Balancing training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Training Gradient Boosting Model...")
model = GradientBoostingClassifier(
    learning_rate=0.02, 
    n_estimators=200, 
    max_depth=5, 
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

y_pred_proba = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-5)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Decision Threshold Found: {optimal_threshold:.4f}")

y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

print("\n--- Final Model Performance ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_optimal))

print("\n--- Top 5 Important Features ---")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(5))

joblib.dump(model, 'churn_model.pkl')

joblib.dump(list(X.columns), 'model_columns.pkl')

print("Model and columns saved successfully to disk!")