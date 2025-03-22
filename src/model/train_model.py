import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import logging
import warnings
import yaml
import pickle
import os 

# Suppress warnings
warnings.filterwarnings("ignore")

# Logging configuration
logger = logging.getLogger("train_model.py")
logger.setLevel(logging.DEBUG)

# File handler for errors
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

# Formatter for log messages
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully from %s", file_path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

import pandas as pd
import os
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into training features and target variable.

    Args:
        df (pd.DataFrame): The input dataset containing features and the target variable.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
        - X_train (pd.DataFrame): Training feature set
        - y_train (pd.Series): Training target variable

    Raises:
        KeyError: If the target column 'price' is missing from the dataset.
        Exception: For any other errors during data splitting.
    """
    try:
        X = df.drop("price", axis=1)
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save test data for prediction step
        os.makedirs("data/processed", exist_ok=True)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        logger.info("Data split into training and test sets")

        return X_train, y_train  # Only returning training data
    except KeyError as e:
        logger.error("Target column 'price' not found in dataset: %s", e)
        raise KeyError("Target column 'price' not found in dataset") from e
    except Exception as e:
        logger.error("Error splitting data: %s", e)
        raise RuntimeError("Unexpected error while splitting data") from e


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting Regressor model.
    """
    try:
        model = GradientBoostingRegressor()
        model.fit(x_train, y_train)
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise

def save_model(model: GradientBoostingRegressor, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.info("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def main() -> None:
    """
    Main execution function for loading data, training the model, and saving it.
    """
    try:
        data = load_data("data/processed/feature_selection.csv")
        x_train, y_train = split_data(data)
        model = train_model(x_train, y_train)
        save_model(model, "models/model.pkl")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# import pandas as pd 
# df=pd.read_csv('data/processed/feature_selection.csv')
# print(df.head())
# print(df.isnull().sum())