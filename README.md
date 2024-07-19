# Credit-approval-pipeline
This repository contains a comprehensive data pipeline for predicting credit card approvals. The project utilizes Prefect for workflow orchestration, generating and processing synthetic data, training machine learning models, and storing the results in various data sources including MySQL, MongoDB, and AWS S3.

![diagrama flujo final](https://github.com/user-attachments/assets/89059b88-3f8d-4a53-98a5-a0ca9a4f5a2d)

**Project Overview**

The credit-approval-pipeline project demonstrates how to build an end-to-end data pipeline for predicting credit card approvals using synthetic data. The pipeline performs the following steps:

Data Generation: Generates synthetic customer data with relevant features such as age, income, credit score, and employment status.
Data Storage: Stores the generated data in MySQL, MongoDB, and AWS S3.
Data Extraction: Extracts data from the various storage systems.
Data Transformation: Combines and transforms the data, including handling categorical variables.
Model Training: Trains a machine learning model (Random Forest) to predict credit card approval based on the customer data.
Model Evaluation: Evaluates the model's performance using metrics like confusion matrix and classification report.
Prediction: Uses the trained model to predict credit approvals for new synthetic data samples.
Visualization: Generates visualizations of the data and model performance, and uploads them to AWS S3.
Key Features
Prefect Orchestration: Utilizes Prefect for managing and orchestrating the data pipeline.
Synthetic Data Generation: Employs the Faker library to create realistic synthetic customer data.
Multi-Source Data Storage: Demonstrates storing data in MySQL, MongoDB, and AWS S3.
Machine Learning: Implements a Random Forest classifier for predicting credit card approvals.
Visualization: Provides data visualizations and model performance metrics.
AWS Integration: Uses AWS S3 for storing data and visualizations.
Repository Structure
pipeline.py: The main script defining the Prefect flow and tasks.
generate_data.py: Script for generating synthetic customer data.
requirements.txt: List of required Python packages.
README.md: Project description and setup instructions.
Getting Started
Prerequisites
Python 3.8 or higher
AWS CLI configured with appropriate credentials
MySQL and MongoDB instances for data storage
Installation

**Output Files**

The pipeline generates the following output files:

data.csv: The synthetic customer data generated for training and evaluation.

combined_data.csv: The combined data extracted from MySQL, MongoDB, and S3.

trained_model.pkl: The trained Random Forest model.

new_data_predictions.csv: Predictions for new synthetic data samples.

data_visualization.png: Visualization of the synthetic data.

confusion_matrix.png: Confusion matrix of the model's performance.
