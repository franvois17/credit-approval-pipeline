import pandas as pd
from pymongo import MongoClient
from generate_data import generate_synthetic_data

def store_data_mongodb(df):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['bank_data']
    collection = db['customer_data']
    collection.insert_many(df.iloc[3000:5000].to_dict('records'))

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data()
    store_data_mongodb(synthetic_data)

