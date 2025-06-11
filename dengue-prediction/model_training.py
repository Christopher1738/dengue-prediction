import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, classification_report)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load preprocessed data"""
    try:
        X = joblib.load('preprocessed_features.pkl')
        y = joblib.load('targets.pkl')
        return X, y
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Run data_preprocessing.py first.")
        exit(1)

def train_models(X, y):
    """Train and evaluate models"""
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Separate targets
    y_cases_train = y_train['next_week_cases']
    y_outbreak_train = y_train['outbreak_risk']
    y_cases_test = y_test['next_week_cases']
    y_outbreak_test = y_test['outbreak_risk']

    # 1. Train Case Prediction Model (XGBoost Regressor)
    print("\nTraining Case Prediction Model...")
    case_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        early_stopping_rounds=10,
        eval_metric='mae'
    )
    
    case_model.fit(
        X_train, y_cases_train,
        eval_set=[(X_test, y_cases_test)],
        verbose=True
    )

    # 2. Train Outbreak Classifier (Gradient Boosting)
    print("\nTraining Outbreak Classifier...")
    outbreak_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    outbreak_model.fit(X_train, y_outbreak_train)

    return case_model, outbreak_model, X_test, y_cases_test, y_outbreak_test

def evaluate_models(case_model, outbreak_model, X_test, y_cases_test, y_outbreak_test):
    """Evaluate and visualize model performance"""
    # Case prediction evaluation
    cases_pred = case_model.predict(X_test)
    mae = np.mean(np.abs(cases_pred - y_cases_test))
    print(f"\nCase Prediction MAE: {mae:.2f}")

    # Outbreak classification evaluation
    outbreak_pred = outbreak_model.predict(X_test)
    outbreak_proba = outbreak_model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_outbreak_test, outbreak_pred))
    
    print(f"Accuracy: {accuracy_score(y_outbreak_test, outbreak_pred):.2f}")
    print(f"F1 Score: {f1_score(y_outbreak_test, outbreak_pred):.2f}")

    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_outbreak_test, outbreak_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title('Outbreak Prediction Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Feature importance
    plot_feature_importance(case_model, X_test.columns)

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Important Features for Case Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def save_models(case_model, outbreak_model):
    """Save trained models"""
    joblib.dump(case_model, 'case_predictor.pkl')
    joblib.dump(outbreak_model, 'outbreak_classifier.pkl')
    print("\nModels saved successfully:")
    print("- case_predictor.pkl")
    print("- outbreak_classifier.pkl")

if __name__ == "__main__":
    print("Loading preprocessed data...")
    X, y = load_data()
    
    print("Training models...")
    case_model, outbreak_model, X_test, y_cases_test, y_outbreak_test = train_models(X, y)
    
    print("\nEvaluating models...")
    evaluate_models(case_model, outbreak_model, X_test, y_cases_test, y_outbreak_test)
    
    save_models(case_model, outbreak_model)