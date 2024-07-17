import pandas as pd
import boto3
from io import StringIO
from generate_data import generate_synthetic_data

def store_data_s3(df, bucket_name, file_name):
    s3_client = boto3.client('s3')
    csv_buffer = StringIO()
    df.iloc[5000:].to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())
    print(f"Data successfully uploaded to {bucket_name}/{file_name}")

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data()
    store_data_s3(synthetic_data, 'fbaquero', 'customer_data.csv')

