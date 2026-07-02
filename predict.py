import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

new_customer_data = pd.DataFrame([{
    'CreditScore': 650,
    'Geography': 'Germany',
    'Gender': 'Female',
    'Age': 45,
    'Tenure': 4,
    'Balance': 95000.00,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 0, 
    'EstimatedSalary': 60000.00
}])

new_customer_processed = pd.get_dummies(new_customer_data)

new_customer_processed = new_customer_processed.reindex(columns=model_columns, fill_value=0)

optimal_threshold = 0.5985
churn_probability = model.predict_proba(new_customer_processed)[:, 1][0]
is_churner = int(churn_probability >= optimal_threshold)

print(f"--- Prediction Results ---")
print(f"Churn Probability: {churn_probability:.2%}")
print(f"High Risk of Leaving: {'YES' if is_churner == 1 else 'NO'}")