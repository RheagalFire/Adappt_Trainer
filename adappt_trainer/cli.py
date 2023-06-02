import argparse
from adappt_trainer.preprocessor import DataProcessor
from adappt_trainer.train_score_deploy import ModelTrainer
from adappt_trainer.inference import ModelInference

def preprocess_data(input_data_dir, output_data_dir):
    processor = DataProcessor(input_data_dir=input_data_dir, output_data_dir=output_data_dir)
    processor.process_data()
    processor.save_data()

def train_model(input_file_path, test_directory, mlflow_logging):
    trainer = ModelTrainer(input_file_path=input_file_path, test_directory=test_directory, mlflow_logging=mlflow_logging)
    trainer.train_model()

def perform_inference(input_file):
    inference = ModelInference(input_file=input_file)
    inference.perform_inference()

def main():
    parser = argparse.ArgumentParser(description='ML Pipeline CLI Tool')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess the data')
    preprocess_parser.add_argument('--input-data-dir', type=str, help='Path to the input data directory')
    preprocess_parser.add_argument('--output-data-dir', default='clean_data',type=str, help='Path to the output data directory')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input-file-path', type=str, help='Path to the input file')
    train_parser.add_argument('--test-directory', default=None,type=str, help='Path to the test data directory')
    train_parser.add_argument('--mlflow-logging', default=False,action='store_true', help='Enable MLflow logging')

    inference_parser = subparsers.add_parser('inference', help='Perform inference')
    inference_parser.add_argument('--input-file', type=str, help='Path to the input file')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_data(args.input_data_dir, args.output_data_dir)
    elif args.command == 'train':
        train_model(args.input_file_path, args.test_directory, args.mlflow_logging)
    elif args.command == 'inference':
        perform_inference(args.input_file)

if __name__ == '__main__':
    main()
