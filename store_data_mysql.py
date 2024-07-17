import pandas as pd
import sqlalchemy
from generate_data import generate_synthetic_data

def store_data_mysql(df):
    engine = sqlalchemy.create_engine('mysql+pymysql://user:password@localhost/bank_data')
    df.iloc[:3000].to_sql('customer_data', engine, index=False, if_exists='replace')

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data()
    store_data_mysql(synthetic_data)


