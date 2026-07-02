import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# 1. Load and prepare data
df = pd.read_csv('Churn_Modelling.csv')
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Balancing data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5]
}

gb_model = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=gb_model, 
    param_grid=param_grid, 
    scoring='recall', 
    cv=3, 
    n_jobs=-1, 
    verbose=2  
)

print("Searching for best parameters (this may take a minute)...")
grid_search.fit(X_train_resampled, y_train_resampled)

print("\n--- TUNING COMPLETE ---")
print(f"Best Parameters Found: {grid_search.best_params_}")