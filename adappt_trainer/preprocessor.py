import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
import logging

class DataProcessor:
    def __init__(self, input_data_dir, output_data_dir='clean_data',preprocessor_path='preprocessor'):
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.preprocessor_path = preprocessor_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())


    def read_data_from_dir(self, directory):
        dataframes = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, filename))
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def process_data(self):
        # Read the input data
        self.logger.info("Reading input data...")
        input_data = self.read_data_from_dir(self.input_data_dir)

        # Check for duplicate entries
        duplicates = input_data.duplicated()
        num_duplicates = duplicates.sum()
        self.logger.info("Number of duplicate entries: %s", num_duplicates)

        # Remove duplicate entries
        input_data = input_data.drop_duplicates()


        # Identify categorical columns
        categorical_cols = input_data.select_dtypes(include=['object']).columns

        # Scale numeric columns (excluding 'exited') using StandardScaler
        numeric_cols = input_data.select_dtypes(include=['float', 'int']).columns
        numeric_cols = [col for col in numeric_cols if col != 'exited']
        scaler = StandardScaler()
        input_data[numeric_cols] = scaler.fit_transform(input_data[numeric_cols])    

        # Encode categorical columns
        le = LabelEncoder()
        for col in categorical_cols:
            input_data[col] = le.fit_transform(input_data[col])
        
        os.makedirs(self.preprocessor_path, exist_ok=True)

        preprocessor_file = os.path.join(self.preprocessor_path,'preprocessor.pkl')
        with open(preprocessor_file, 'wb') as f:
            pickle.dump((le,scaler), f)
        
        self.logger.info("Data processing completed.")
        self.input_data = input_data
    def save_data(self):
        # Save the cleaned training data to persistent storage
        os.makedirs(self.output_data_dir, exist_ok=True)
        cleaned_data_path = os.path.join(self.output_data_dir, 'cleaned_training_data.csv')
        self.input_data.to_csv(cleaned_data_path, index=False)
        self.logger.info("Cleaned training data saved to %s", cleaned_data_path)

