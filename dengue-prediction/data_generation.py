# data_generation.py
import pandas as pd
import numpy as np

def create_dengue_dataset(size=1000):
    """Generate synthetic dengue dataset"""
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(30, 2, size),
        'rainfall': np.random.gamma(3, 2, size),
        'humidity': np.random.uniform(60, 95, size),
        'vegetation_index': np.random.normal(0.6, 0.1, size),
        'population_density': np.random.poisson(5000, size),
        'past_cases': np.random.poisson(50, size),
        'week_of_year': np.random.randint(1, 53, size)
    }
    
    # Simulate relationship with dengue cases
    cases = (
        0.3 * data['temperature'] + 
        0.4 * data['rainfall'] + 
        0.2 * data['humidity'] + 
        0.1 * data['vegetation_index'] +
        0.05 * (data['population_density'] / 1000) +
        0.8 * data['past_cases'] +
        np.random.normal(0, 10, size)
    )
    
    data['next_week_cases'] = np.clip(cases, 0, None).astype(int)
    data['outbreak_risk'] = (data['next_week_cases'] > 100).astype(int)
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_dengue_dataset(5000)
    df.to_csv('dengue_data.csv', index=False)
    print("Dataset created with shape:", df.shape)
    print("Outbreak distribution:\n", df['outbreak_risk'].value_counts(normalize=True))