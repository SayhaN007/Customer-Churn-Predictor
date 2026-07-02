import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_model()

OPTIMAL_THRESHOLD = 0.5985 

st.title("📉 Customer Churn Predictor")
st.markdown("Enter a customer's details below to instantly predict their likelihood of leaving the service.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': salary
    }])

    input_processed = pd.get_dummies(input_data)
    input_processed = input_processed.reindex(columns=model_columns, fill_value=0)

    churn_probability = model.predict_proba(input_processed)[:, 1][0]
    
    st.divider()
    st.subheader("Prediction Results")
    
    if churn_probability >= OPTIMAL_THRESHOLD:
        st.error(f"🚨 **HIGH RISK: This customer is likely to churn.**")
    else:
        st.success(f"✅ **LOW RISK: This customer is likely to stay.**")
        
    st.metric(label="Calculated Churn Probability", value=f"{churn_probability:.1%}")
    st.progress(float(churn_probability))