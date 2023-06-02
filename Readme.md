# Adappt_Trainer

Adappt_Trainer is a Python package that provides data preprocessing, model training, and inference functionalities for an attrition risk prediction model. It offers a CLI (Command-Line Interface) to perform these operations conveniently.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Running CLI Commands using Docker](#running-cli-commands-using-docker)
-[Notebook](#notebook)

## Installation

To use the `Adappt_Trainer` package, you can follow these steps:

1. Clone the repository:
```
git clone https://github.com/RheagalFire/Adappt_Trainer.git

```
2. Navigate to the project directory
```
cd Adappt_Trainer

```
3. Create a virtual environment:

```
python3 -m venv env
source env/bin/activate
```
4.Install the package using setup.py
```
python setup.py install

```
This will install the Adappt_Trainer package and its dependencies.

## Usage
### CLI Commands 
The Adappt_Trainer package provides various CLI commands to perform data preprocessing, model training, and inference. Here are the available commands:
- Preprocess: Preprocesses the input data to prepare it for training. Example:
```
ml_pipeline preprocess --input-data-dir /path/to/input_data

```
- Train: Trains the attrition risk prediction model using the preprocessed data.While Training pass in the flag --mlflow-logging to log metrics,artifacts using MLflow Example:
```
ml_pipeline train --input-file-path /path/to/preprocessed_data.csv

```
- Infer: Performs inference on new data using the trained model. Example:
```
ml_pipeline infer --input-file /path/to/inference_data.csv 

```
### Running CLI Commands using Docker
The Adappt_Trainer package can also be run using Docker. Here's how you can do it:

1. Build the Docker image
```
docker build -t adappt_trainer .

```
2.Run the Docker container and execute CLI commands
```
docker run -v /path/to/data:/app/data -it adappt_trainer [command] [options]

```
- Example : 
    ```
    docker run -v /path/to/data:/app/data -it adappt_trainer preprocess --input-data-dir /app/data/input_data

    ```
## Notebook
You can checkout the notebook file on how to preprocess,train,infer using the modules [here](notebooks/Running%20Modules.ipynb)  






