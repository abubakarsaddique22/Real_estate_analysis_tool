import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os

# Logging configuration
logger = logging.getLogger("data_pipeline")
logger.setLevel(logging.DEBUG)

# File handler for errors
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

# Formatter for log messages
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_model(file_path):
    """
    Load the trained model from a file.

    Args:
        file_path: Path to the model file.

    Returns:
        Loaded model.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_test_data():
    """
    Load test data from saved CSV files.

    Returns:
        X_test: Test feature set as a DataFrame.
        y_test: Test target variable as a Series.
    """
    try:
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()  # Convert to Series
        logger.debug("Test data loaded successfully")
        return X_test, y_test
    except FileNotFoundError:
        logger.error("Test data files not found")
        raise
    except Exception as e:
        logger.error("Error loading test data: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the regression model and return evaluation metrics.

    Args:
        model: Trained model.
        X_test: Test feature set.
        y_test: Test target values.

    Returns:
        Dictionary with model evaluation metrics (MAE, MSE, R² Score).
    """
    try:
        y_pred = model.predict(X_test)

        # Regression Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_dict = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "r2_score": r2
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def save_metrics(metrics, file_path):
    """
    Save the evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary containing evaluation metrics.
        file_path: Path to save the JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """
    Main function to load model, evaluate on test data, and save metrics.
    """
    try:
        clf = load_model('./models/model.pkl')
        X_test, y_test = load_test_data()  # Load test data

        metrics = evaluate_model(clf, X_test, y_test)  # Evaluate model
        
        save_metrics(metrics, 'reports/metrics.json')  # Save metrics
        
        print("Model Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")  # Print metrics with 4 decimal places
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
