# 📉 Customer Churn Prediction & Web App

## 📌 Project Overview
This project is an end-to-end Machine Learning solution designed to predict customer churn for a subscription-based service or bank. By identifying high-risk customers before they leave, businesses can proactively offer retention incentives, saving significant revenue.

The project features a highly tuned **Gradient Boosting Classifier**, techniques to handle highly imbalanced data (**SMOTE**), dynamic decision thresholding, and a fully interactive web application built with **Streamlit**.

---

## 🚀 Key Features & Methodologies
* **Data Preprocessing:** One-hot encoding for categorical variables and strategic feature selection.
* **Class Imbalance Handling:** Implemented SMOTE (Synthetic Minority Over-sampling Technique) strictly on the training data to prevent data leakage and improve minority class detection.
* **Hyperparameter Tuning:** Automated `GridSearchCV` to find the optimal learning rate, tree depth, and estimator count.
* **Dynamic Thresholding:** Calculated the mathematical Precision-Recall curve to find the exact probability threshold (0.5985) that maximizes the F1-Score, balancing false alarms with missed churners.
* **Interactive UI:** A real-time web application where non-technical users can input customer data and instantly receive a churn risk probability.

---

## 📊 Model Performance
* **Algorithm:** Gradient Boosting Classifier
* **ROC-AUC Score:** 0.858
* **Business Impact:** Successfully optimized to catch the majority of actual churners while maintaining a highly reliable precision rate, making it immediately deployable for targeted marketing campaigns.
* **Top Predictive Features:**
  1. Age
  2. Active Member Status
  3. Number of Products
  4. Account Balance

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`
* **Frontend:** `streamlit`

---

## 📂 File Structure
* `tune.py` - Runs GridSearch to find the best model parameters. (Run once).
* `main.py` - Trains the final model using SMOTE and optimal parameters, calculates the best threshold, and exports `.pkl` files.
* `app.py` - The Streamlit web application script.
* `Churn_Modelling.csv` - The raw dataset.
* `churn_model.pkl` & `model_columns.pkl` - Serialized model and column structures for deployment.

---

## 💻 Installation & Usage

### 1. Clone the repository and install dependencies
```bash
git clone [https://github.com/yourusername/customer-churn-predictor.git](https://github.com/yourusername/customer-churn-predictor.git)
cd customer-churn-predictor
python -m pip install -r requirements.txt
