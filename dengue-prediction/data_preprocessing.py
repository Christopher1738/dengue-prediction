import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(filepath):
    """Load and preprocess data"""
    df = pd.read_csv(filepath)
    
    # Feature engineering
    df['rain_temp_interaction'] = df['rainfall'] * df['temperature']
    df['pop_density_sqrt'] = np.sqrt(df['population_density'])
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year']/52)
    
    # Select features and targets - FIXED: Using list instead of set for column selection
    features = df[[
        'temperature', 'rainfall', 'humidity', 'vegetation_index',
        'population_density', 'past_cases', 'rain_temp_interaction',
        'pop_density_sqrt', 'week_sin', 'week_cos'
    ]]
    targets = df[['next_week_cases', 'outbreak_risk']]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    
    # Normalize features - FIXED: Preserve column names
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_imputed)
    
    # Save artifacts
    joblib.dump(scaler, 'scaler.pkl')
    
    # Return as DataFrames to preserve column names
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
    return scaled_features_df, targets

if __name__ == "__main__":
    try:
        X, y = preprocess_data('dengue_data.csv')
        print("Preprocessed features shape:", X.shape)
        print("Targets shape:", y.shape)
        
        # Save processed data for next steps
        joblib.dump(X, 'preprocessed_features.pkl')
        joblib.dump(y, 'targets.pkl')
        print("Data saved to preprocessed_features.pkl and targets.pkl")
        
    except FileNotFoundError:
        print("Error: dengue_data.csv not found. Please run data_generation.py first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")