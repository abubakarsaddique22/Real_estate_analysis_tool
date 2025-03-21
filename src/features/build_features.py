import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# import shap
from typing import Tuple
import logging

import warnings
warnings.filterwarnings("ignore")
import logging
import yaml

# Logging configuration
logger = logging.getLogger('build_features')
logger.setLevel(logging.DEBUG)  # Capture all levels but only store errors

# File handler (captures ERROR and above)
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

# Formatter for the logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add only the file handler
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:

    """
    Load parameters from a YAML file.
    
    """

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def drop_columns(df: pd.DataFrame, columns:list) -> pd.DataFrame:

    """
    Loads a CSV file and drops unnecessary columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to drop.
    
    Returns:
        pd.DataFrame: DataFrame with columns dropped.
    
    """
    
    try:
        df = df.drop(columns=columns,axis=1)
        logger.info('Columns dropped successfully')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    Encodes categorical features using OrdinalEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]])
            # print(f"Encoded categories for {col}: {oe.categories_}")
        return df
    except Exception as e:
        logging.error("Error encoding categorical features: %s", e)
        raise


def main():
    try:
        df = pd.read_csv('data/processed/imputed_data.csv')

        # Load and clean data
        """
        just drop the column becuase I already identify which columns important using these 
        technique and these code in notebook that notebook name feature_selection.ipynb

        Here are the feature selection techniques used in code:
        Correlation-Based Feature Selection
        Random Forest Feature Importance
        Gradient Boosting Feature Importance
        Permutation Feature Importance
        LASSO Regression Feature Selection
        Recursive Feature Elimination (RFE)
        Linear Regression Coefficients
        SHAP (SHapley Additive Explanations) Feature Importance
        RFE--> it is tree model and most importand technique for feature importance
        
        """
        df=drop_columns(df,['society','price_per_sqft','Location','area_room_ratio','Purpose'])

        # Encode categorical features
        df = encode_categorical_features(df)
        

        # save data
        df.to_csv('data/processed/feature_selection.csv')


    except Exception as e:
        logging.error("Error in main function: %s", e)
        raise

if __name__ == '__main__':
    main()
