import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import mlflow
import os
import logging

class ModelTrainer:
    def __init__(self, input_file_path,output_path='model_artifacts',preprocessor_path='preprocessor',test_directory=None,split_percentage=0.2,mlflow_logging=False):
        self.file_path = input_file_path
        self.split_percentage = split_percentage
        self.output_path = output_path
        self.test_directory = test_directory
        self.preprocessor_path = preprocessor_path
        self.mlflow_logging = mlflow_logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
    def train_model(self):
        # Read the cleaned training data
        for filename in os.listdir(self.file_path):
            if filename.endswith('.csv'):
                data = pd.read_csv(os.path.join(self.file_path, filename))

        # Split the data into features and target
        X = data.drop('exited', axis=1)
        y = data['exited']
        if(self.test_directory):
            X_train = X
            y_train = y
            for filename in os.listdir(self.test_directory):
                if filename.endswith('.csv'):
                    test_data = pd.read_csv(os.path.join(self.test_directory, filename))
            le,scaler = pickle.load(open(os.path.join(self.preprocessor_path,'preprocessor.pkl'),'rb'))
            categorical_cols = test_data.select_dtypes(include=['object']).columns
            numeric_cols = test_data.select_dtypes(include=['float', 'int']).columns
            numeric_cols = [col for col in numeric_cols if col != 'exited']
            test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
            for col in categorical_cols:
                test_data[col] = le.fit_transform(test_data[col])
            X_test = test_data.drop('exited', axis=1)
            y_test = test_data['exited']
        else:
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_percentage, random_state=42)

        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.logger.info("Model training completed.")
        # Score the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        os.makedirs(self.output_path, exist_ok=True)
        # Write the model to persistent storage
        model_path = os.path.join(self.output_path,'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Write the scoring metrics to the specified output path
        metrics_path = os.path.join(self.output_path,'scoring_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write('Classification Report:\n')
            f.write(report)

        if(self.mlflow_logging):
            with mlflow.start_run():
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_artifact(model_path)
                    mlflow.log_artifact(metrics_path)
                    mlflow.log_text(report, "classification_report")

        

