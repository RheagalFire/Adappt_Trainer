import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle 
import logging

class ModelInference:
    def __init__(self, input_file,model_path='model_artifacts', output_file_path = 'inference_metrics',preprocessor_path='preprocessor'):
        self.model_path = model_path
        self.input_file = input_file
        self.output_file_path = output_file_path
        self.preprocessor_path=preprocessor_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def perform_inference(self):
        # Load the input data
        for filename in os.listdir(self.input_file):
            if filename.endswith('.csv'):
                data = pd.read_csv(os.path.join(self.input_file, filename))
        # Load the model file
        model = pickle.load(open(os.path.join(self.model_path,'model.pkl'),'rb'))
        # Load the preprocessor
        le,scaler = pickle.load(open(os.path.join(self.preprocessor_path,'preprocessor.pkl'),'rb'))
        # Perform the preprocessing
        categorical_cols = data.select_dtypes(include=['object']).columns
        numeric_cols = data.select_dtypes(include=['float', 'int']).columns
        numeric_cols = [col for col in numeric_cols if col != 'exited']
        data[numeric_cols] = scaler.transform(data[numeric_cols])
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

        # Generate predictions
        predictions = model.predict(data)

        # Add the predictions to the data
        data['predictions'] = predictions
        # Make the directory if not already present
        os.makedirs(self.output_file_path, exist_ok=True)
        # Write the data with predictions to the output file
        output_file = os.path.join(self.output_file_path,'inference.csv')
        data.to_csv(output_file, index=False)

        self.logger.info("Inference completed.")