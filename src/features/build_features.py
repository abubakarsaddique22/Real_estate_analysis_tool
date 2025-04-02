import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle


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

# def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    """
    Encodes categorical features using OrdinalEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    try:
        categorical_cols = df.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]])
            # print(f"Encoded categories for {col}: {oe.categories_}")
        return df
    except Exception as e:
        logging.error("Error encoding categorical features: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the dataset using a pipeline."""
    try:
        numeric_features = ['Parking Spaces', 'Bedrooms', 'Bathrooms', 'Servant Quarters', 'Kitchens', 'Store Rooms', 'area']
        categorical_ordinal_features = ['property Type']
        categorical_onehot_features = ['Age Possession','colony','province','City']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal_features),
                ('cat1', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_onehot_features)
            ],
            remainder='passthrough'  # Keeps other columns like 'price'
        )

        pipeline = Pipeline([('preprocessor', preprocessor)])

        # Fit and transform the data
        df_transformed = pipeline.fit_transform(df)

        # Extract column names dynamically
        onehot_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat1'].get_feature_names_out(categorical_onehot_features)

        # Get original columns that were passed through
        passthrough_columns = [col for col in df.columns if col not in (numeric_features + categorical_ordinal_features + categorical_onehot_features)]

        # Create final column names
        transformed_columns = numeric_features + categorical_ordinal_features + list(onehot_feature_names) + passthrough_columns

        # Convert to DataFrame with correct columns
        df_transformed = pd.DataFrame(df_transformed, columns=transformed_columns)

        return df_transformed

    except Exception as e:
        logger.error("Error in preprocessing data: %s", e)
        raise



def data_type_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change the data types of specific columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with updated data types.
    """
    try:
        df['City'] = df['City'].astype('category')
        df['property Type'] = df['property Type'].astype('category')
        df['colony'] = df['colony'].astype('category')
        df['province'] = df['province'].astype('category')
        df['Parking Spaces'] = df['Parking Spaces'].astype('int')
        df['Age Possession'] = df['Age Possession'].astype('category')
        return df
    except Exception as e:
        logging.error("Error changing data types: %s", e)
        raise


def main():
    try:
        df = pd.read_csv('data/processed/imputed_data.csv')
        print(df.shape)
        # Load and clean data
    #     """
    #     just drop the column becuase I already identify which columns important using these 
    #     technique and these code in notebook that notebook name feature_selection.ipynb

    #     Here are the feature selection techniques used in code:
    #     Correlation-Based Feature Selection
    #     Random Forest Feature Importance
    #     Gradient Boosting Feature Importance
    #     Permutation Feature Importance
    #     LASSO Regression Feature Selection
    #     Recursive Feature Elimination (RFE)
    #     Linear Regression Coefficients
    #     SHAP (SHapley Additive Explanations) Feature Importance
    #     RFE--> it is tree model and most importand technique for feature importance
        
    #     """
        df=drop_columns(df,['society','price_per_sqft','Location','area_room_ratio','Purpose'])

        # just upper and lower case in property type column covert to Houses 
        df['property Type'].replace({'Upper':"Houses",'Lower':"Houses"},inplace=True)
        print(df['property Type'].value_counts())
        df=data_type_change(df)
       
        preprocessor = preprocess_data(df)
        print(df.shape)
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    #     # save data
        preprocessor.to_csv('data/processed/feature_selection.csv',index=False)

    except Exception as e:
        logging.error("Error in main function: %s", e)
        raise
    # Get transformed data and preprocesso

if __name__ == '__main__':
    main()
