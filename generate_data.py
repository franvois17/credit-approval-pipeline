import pandas as pd
import numpy as np
from faker import Faker
import random

def generate_synthetic_data(num_samples=7000, income_type='annual'):
    fake = Faker()
    data = {
        'customer_id': [fake.uuid4() for _ in range(num_samples)],
        'name': [fake.name() for _ in range(num_samples)],
        'age': np.random.randint(18, 70, num_samples),
        'income': np.random.randint(20000, 120000, num_samples) if income_type == 'annual' else np.random.randint(2000, 10000, num_samples),
        'credit_score': np.random.randint(300, 850, num_samples),
        'employment_status': [random.choice(['Employed', 'Unemployed', 'Self-employed', 'Student']) for _ in range(num_samples)],
        'existing_debt': np.random.randint(0, 50000, num_samples),
        'has_defaulted': np.random.randint(0, 2, num_samples),
        'is_approved': np.random.randint(0, 2, num_samples),
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data()
    print(synthetic_data.head())

