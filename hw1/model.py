import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, classification_report, ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# ==================== 0. Data Loading ====================
file_path = "E:/diabetes.csv"
df = pd.read_csv(file_path)

# ==================== 1. Data Cleaning ====================
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

# Print missing values after replacement
print("Missing Value Statistics:\n", df.isnull().sum())

# Standardization before KNN imputation
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Outcome'])), columns=df.columns[:-1])

# Impute missing values with KNN
imputer = KNNImputer(n_neighbors=3, weights="distance")
df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns[:-1])
df_imputed["Outcome"] = df["Outcome"].astype(int)  # Recover target dtype

# ==================== 2. Feature Engineering ====================
X = df_imputed.drop(columns=["Outcome"])
y = df_imputed["Outcome"]

# Feature selection
selector = SelectKBest(f_classif, k=6)
X_selected = selector.fit_transform(X, y)
selected_cols = df.columns[:-1][selector.get_support()]
print("\nSelected Features:", selected_cols.tolist())

# ==================== 3. Handle Class Imbalance ====================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# ==================== 4. Model Training & Tuning ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, param_grid, model_name):
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='recall'  # Optimize for recall
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Model Evaluation
    y_pred = best_model.predict(X_test)
    
    print(f"\nModel: {model_name}")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Recall Score: {recall_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Confusion Matrix (Test Set) - {model_name}")
    plt.show()
    
    # Feature Importance Analysis
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        plt.figure(figsize=(10, 4))
        sns.barplot(x=importance, y=selected_cols, palette="viridis")
        plt.title(f"Feature Importance Ranking - {model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()
    
    return best_model, recall_score(y_test, y_pred)

# Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
gb_model, gb_recall = train_and_evaluate_model(GradientBoostingClassifier(random_state=42), gb_param_grid, "Gradient Boosting")

# Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5]
}
rf_model, rf_recall = train_and_evaluate_model(RandomForestClassifier(random_state=42), rf_param_grid, "Random Forest")

# XGBoost
# XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
xgb_model, xgb_recall = train_and_evaluate_model(
    XGBClassifier(random_state=42, eval_metric='logloss'), 
    xgb_param_grid, 
    "XGBoost"
)

# ==================== 5. Compare Model Results ====================
print("\nModel Recall Scores:")
print(f"Gradient Boosting: {gb_recall:.4f}")
print(f"Random Forest: {rf_recall:.4f}")
print(f"XGBoost: {xgb_recall:.4f}")

