import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import logging
import warnings
import yaml
import pickle
import os
import json
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

# Logging configuration
logger = logging.getLogger("model_pipeline")
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
    Load parameters from a YAML configuration file.

    Args:
        params_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary of parameters loaded from the YAML file.
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
    Load a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
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

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Input dataset containing features and target.

    Returns:
        Tuple: (X_train, y_train, X_test, y_test)
    """

    try:
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        X = df.drop("price", axis=1)
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Data split into training and test sets")
        return X_train, y_train, X_test, y_test  # Returning both training and test data
    except KeyError as e:
        logger.error("Target column 'price' not found in dataset: %s", e)
        raise KeyError("Target column 'price' not found in dataset") from e
    except Exception as e:
        logger.error("Error splitting data: %s", e)
        raise RuntimeError("Unexpected error while splitting data") from e

def create_pipeline() -> Pipeline:
    """
    Create a machine learning pipeline including preprocessing and the XGBoost model.

    Returns:
        Pipeline: Configured sklearn Pipeline object.
    """
    try:
        numeric_features = ['parking_spaces', 'Bedrooms', 'Bathrooms', 'servant_Quarters', 'Kitchens', 'store_rooms', 'area']
        categorical_ordinal_features = ['property_type']
        categorical_onehot_features = ['age_possession','colony','province','City']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal_features),
                ('cat1', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_onehot_features)
            ],
            remainder='passthrough'
        )

        # export the preprocessor.pkl file 
        

        model_pipeline  = Pipeline([('preprocessor', preprocessor),
                             ('model', XGBRegressor(min_child_weight=2, 
                                                    n_estimators=899, 
                                                    gamma=0.9717292730761442, 
                                                    reg_alpha=3.3278429742954385, 
                                                    learning_rate=0.054860581080248355, 
                                                    max_depth=12, 
                                                    reg_lambda=0.06187338470330408, 
                                                    subsample=0.5923264145395795, 
                                                    colsample_bytree=0.9139276831537941))  # XGBoost Model                        
                             
                             ])
        
        

        return model_pipeline 
    except Exception as e:
        logger.error("Error in preprocessing data: %s", e)
        raise

def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train the pipeline model using K-Fold cross-validation.

    Args:
        x_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.

    Returns:
        Pipeline: Trained model pipeline.
    """
        
    try:
        model_pipeline = create_pipeline()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model_pipeline, x_train, y_train, cv=kfold, scoring='r2')

        logger.info(f"Mean R² Score (Training Set): {scores.mean():.4f}")
        print(f"Mean R² Score (Training Set): {scores.mean():.4f}")

        model_pipeline.fit(x_train, y_train)
        logger.info("Model trained successfully")
        return model_pipeline
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on test data and return metrics.

    Args:
        model (Pipeline): Trained model pipeline.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Actual test target values.

    Returns:
        dict: Evaluation metrics (MAE, MSE, R²).
    """
    try:
        y_pred = model.predict(X_test)

        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
       # n = number of observations, k = number of predictors
        n = len(y_test)
        k = X_test.shape[1]

        # Adjusted R²
        adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - k - 1))
        metrics_dict = {
            # "mean_absolute_error": mae,
            # "mean_squared_error": mse,
            "r2_score": r2,
            "adjusted_r2":adjusted_r2
        }

        logger.info("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def save_model(model: Pipeline, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.

    Args:
        model (Pipeline): Trained model.
        file_path (str): Path to save the model file.
    """
      
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.info("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save model evaluation metrics to a JSON file.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
        file_path (str): Path to save the JSON file.
    """
        
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """
    Main execution pipeline:
    - Load data
    - Split data
    - Train model
    - Save model
    - Evaluate model
    - Save evaluation metrics
    """
       
    try:
        # Load the data
        data = load_data("data/processed/feature_selection.csv")
        # print(f"Data shape: {data.shape}")
        # print(data.columns)

        # Split the data
        X_train, y_train, X_test, y_test = split_data(data)

        # Train the model
        model = train_model(X_train, y_train)

        # Save the model
        save_model(model, "models/model.pkl")

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the metrics
        save_metrics(metrics, "reports/metrics.json")

        # Print metrics
        print("Model Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    except Exception as e:
        logger.error('Pipeline execution failed: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
