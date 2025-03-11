import os 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import seaborn as sns 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")
import logging
import yaml

# Logging configuration
logger = logging.getLogger('impute_and_outlier')
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

    parameters:
    ----------
    params_path :str
        The input str from path .

    Returns
    -------
    dict
        return dict 
    
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

def outlier_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:

    """
    Remove outliers from the price_per_sqft column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.
    
    """

    try:
        Q1 = df['price_per_sqft'].quantile(0.25)
        Q3 = df['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

        outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x: x * 9 if x < 1000 else x)
        outliers_sqft['price_per_sqft'] = round((outliers_sqft['price'] * 10000000) / outliers_sqft['area'])

        df.update(outliers_sqft)
        return df
    except Exception as e:
        logger.error("Error in outlier_price_per_sqft: %s", e)
        raise

def outlier_area(df: pd.DataFrame) -> pd.DataFrame:

    """
    Remove outliers from the 'area' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """
        
    try:
        return df[df['area'] <= 12000]
    except Exception as e:
        logger.error("Error in outlier_area: %s", e)
        raise

def bedroom_outliers(df: pd.DataFrame) -> pd.DataFrame:

    """
    Remove outliers from the 'bedrooms' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """
    
    try:
        return df[df['Bedrooms'] <= 10]
    except Exception as e:
        logger.error("Error in bedroom_outliers: %s", e)
        raise

def bathrooms_outlier(df: pd.DataFrame) -> pd.DataFrame:
         
    """
    Remove outliers from the 'bathrooms' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """
         
    try:
        return df[df['Bathrooms'] <= 10]
    except Exception as e:
        logger.error("Error in bathrooms_outlier: %s", e)
        raise

def kitchen_outlier(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Remove outliers from the 'kitchen' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """

    try:
        if df is None or df.empty:
            logger.error("DataFrame is empty in kitchen_outlier.")
            raise ValueError("DataFrame is empty.")

        if 'Kitchens' not in df.columns:
            logger.error("Column 'Kitchens' not found in DataFrame.")
            raise KeyError("Column 'Kitchens' not found in DataFrame.")

        return df[df['Kitchens'] <= 3]
    except Exception as e:
        logger.error("Error in kitchen_outlier: %s", e)
        raise

def Area_with_bedroom_outlier(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Remove outliers from the 'Area_with_bedroom' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """
         
    try:
        df['price_per_sqft'] = round((df['price'] * 10000000) / df['area'])
        df['area_room_ratio'] = df['area'] / df['Bedrooms']
        df = df[df['area_room_ratio'] > 100]
        df = df[df['Bedrooms'] != 10]
        return df
    except Exception as e:
        logger.error("Error in Area_with_bedroom_outlier: %s", e)
        raise

def parking_spaces_outlier(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Remove outliers from the 'parking spaces' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the outliers removed.

    """
       
    try:
        Q1 = df["Parking Spaces"].quantile(0.25)
        Q3 = df["Parking Spaces"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return df[(df["Parking Spaces"] >= lower_bound) & (df["Parking Spaces"] <= upper_bound)]
    except Exception as e:
        logger.error("Error in parking_spaces_outlier: %s", e)
        raise

def main():
    try:
        df = pd.read_csv('data/processed/data_version_preprocessed.csv')

        if df is None or df.empty:
            logger.error("Loaded DataFrame is empty.")
            raise ValueError("Loaded DataFrame is empty.")

        df = outlier_price_per_sqft(df)
        df = outlier_area(df)
        df = bedroom_outliers(df)
        df = bathrooms_outlier(df)
        df = kitchen_outlier(df)
        df = Area_with_bedroom_outlier(df)
        df = parking_spaces_outlier(df)

        df.to_csv('data/processed/outlier_remove.csv', index=False)
        logger.info("Outlier removal process completed successfully.")

    except Exception as e:
        logger.error("Error occurred during processing: %s", e)
        raise

if __name__ == '__main__':
    main()