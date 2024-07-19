from prefect import flow, task
import pandas as pd
import sqlalchemy
from pymongo import MongoClient
import boto3
from io import StringIO, BytesIO
import os
from generate_data import generate_synthetic_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib  # Para guardar y cargar el modelo

@task
def clean_up_files():
    files = ["data.csv", "combined_data.csv", "trained_model.pkl", "new_data_predictions.csv"]
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"{file} has been deleted.")
        else:
            print(f"{file} does not exist.")

@task
def generate_data(num_samples=7000, file_path='data.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print("Data loaded from file.")
    else:
        df = generate_synthetic_data(num_samples=num_samples)
        df.to_csv(file_path, index=False)
        print("Data generated and saved to file.")
    return df

@task
def store_data_mysql(df):
    engine = sqlalchemy.create_engine('mysql+pymysql://user:password@localhost/bank_data')
    df.iloc[:3000].to_sql(name='customer_data', con=engine, if_exists='replace', index=False)

@task
def store_data_mongodb(df):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['bank_data']
    collection = db['customer_data']
    collection.insert_many(df.iloc[3000:5000].to_dict('records'))

@task
def store_data_s3(df, bucket_name, file_name):
    s3_client = boto3.client('s3')
    csv_buffer = StringIO()
    df.iloc[5000:].to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())
    print(f"Data successfully uploaded to {bucket_name}/{file_name}")

@task
def extract_data_mysql():
    engine = sqlalchemy.create_engine('mysql+pymysql://user:password@localhost/bank_data')
    df = pd.read_sql('SELECT * FROM customer_data', con=engine)
    return df

@task
def extract_data_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['bank_data']
    collection = db['customer_data']
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

@task
def extract_data_s3(bucket_name, file_name):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(BytesIO(obj['Body'].read()))
    print(f"Data successfully extracted from {bucket_name}/{file_name}")
    return df

@task
def combine_data(df1, df2, df3):
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    return combined_df

@task
def transform_data(df):
    df = pd.get_dummies(df, columns=['employment_status'], drop_first=True)
    return df

@task
def inspect_data(df):
    print(df.info())
    print(df.head())
    return df

@task
def select_numeric_columns(df):
    df_numeric = df.select_dtypes(include=[float, int])
    print("Selected numeric columns:")
    print(df_numeric.head())
    return df_numeric

@task
def save_combined_data(df, file_path='combined_data.csv'):
    df.to_csv(file_path, index=False)
    print(f"Combined data saved to {file_path}")

@task
def train_model(df, model_path='trained_model.pkl'):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    X = df.drop('is_approved', axis=1)
    y = df['is_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Guardar el modelo entrenado
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")
    
    return model, X_test, y_test, y_pred

@task
def upload_model_to_s3(bucket_name, file_name, model_path='trained_model.pkl'):
    s3_client = boto3.client('s3')
    with open(model_path, "rb") as f:
        s3_client.upload_fileobj(f, bucket_name, file_name)
    print(f"Model successfully uploaded to {bucket_name}/{file_name}")

@task
def visualize_data(df, bucket_name, file_name):
    plt.figure()
    sns.pairplot(df[['age', 'income', 'credit_score', 'existing_debt', 'has_defaulted', 'is_approved']])
    plot_path = "data_visualization.png"
    plt.savefig(plot_path)
    plt.close()
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(plot_path, bucket_name, file_name)
    print(f"Data visualization uploaded to {bucket_name}/{file_name}")

@task
def plot_confusion_matrix(y_test, y_pred, bucket_name, file_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    plt.figure()
    disp.plot()
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(plot_path, bucket_name, file_name)
    print(f"Confusion matrix uploaded to {bucket_name}/{file_name}")

@task
def predict_new_data(model_path='trained_model.pkl'):
    # Cargar el modelo guardado
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Generar 5 nuevos datos aleatorios para predicción
    new_data = generate_synthetic_data(num_samples=5)
    print("New data generated for prediction:")
    print(new_data)
    
    # Transformar los datos nuevos de la misma manera que los datos originales
    new_data_transformed = pd.get_dummies(new_data, columns=['employment_status'], drop_first=True)
    
    # Asegurarse de que las columnas coincidan con las del entrenamiento
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in new_data_transformed.columns:
            new_data_transformed[col] = 0  # Añadir columna faltante con valor 0

    new_data_transformed = new_data_transformed[expected_columns]  # Reordenar columnas para que coincidan
    
    # Predecir la aprobación con el modelo cargado
    predictions = model.predict(new_data_transformed)
    new_data['is_approved'] = predictions
    print("Predictions for new data:")
    print(new_data)
    
    return new_data

@task
def confirm_files_exist():
    files = ["data.csv", "combined_data.csv", "trained_model.pkl", "new_data_predictions.csv"]
    for file in files:
        if os.path.exists(file):
            print(f"{file} exists.")
        else:
            print(f"{file} does not exist.")

@flow(name="Credit Card Offer Pipeline")
def credit_card_offer_pipeline():
    clean_up_files()
    
    data = generate_data(num_samples=7000, file_path='data.csv')
    store_data_mysql(data)
    store_data_mongodb(data)
    store_data_s3(data, 'fbaquero', 'customer_data.csv')
    
    data_mysql = extract_data_mysql()
    data_mongodb = extract_data_mongodb()
    data_s3 = extract_data_s3('fbaquero', 'customer_data.csv')
    
    combined_data = combine_data(data_mysql, data_mongodb, data_s3)
    
    transformed_data = transform_data(combined_data)
    inspected_data = inspect_data(transformed_data)
    numeric_data = select_numeric_columns(inspected_data)
    
    save_combined_data(combined_data, file_path='combined_data.csv')
    
    visualize_data(numeric_data, 'fbaquero', 'data_visualization.png')
    
    model, X_test, y_test, y_pred = train_model(numeric_data)
    
    # Nueva tarea para subir el modelo a S3
    upload_model_to_s3('fbaquero', 'trained_model.pkl')
    
    plot_confusion_matrix(y_test, y_pred, 'fbaquero', 'confusion_matrix.png')
    
    new_data_predictions = predict_new_data()
    new_data_predictions.to_csv('new_data_predictions.csv', index=False)
    print("New data predictions saved to new_data_predictions.csv")
    
    # Confirmar la existencia de los archivos generados
    confirm_files_exist()

if __name__ == "__main__":
    credit_card_offer_pipeline()


