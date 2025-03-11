import os 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

import warnings
warnings.filterwarnings("ignore")
import logging
import yaml

# Logging configuration
logger = logging.getLogger('data_ingestion')
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
    Drop columns from a DataFrame.

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


def clean_price(df: pd.DataFrame, column) -> pd.DataFrame:
    
    """
    Clean the price column.

    Args:
       column,df (DataFrame),: Input dataframe and column pass the price column

    Returns:
       pd.DataFrame: Valid price or NaN if invalid.
    
    """
    
    try:
        if type(column) == float:
            return column
        else:
            if column[2] == 'Lakh':
                return round(float(column[1])/100, 2)
            else:
                return round(float(column[1]), 2)
        logger.info('Price column cleaned successfully')
        return df
            
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# Get the current year
current_year: int = datetime.now().year

# Function to clean and convert built year column
def clean_built_year(value: int | float | str | None) -> int | float:
    
    """
    Cleans and converts the built year into a valid integer year.

    Args:
        value (int, float, str, None): Input year (can be numeric or string).

    Returns:
        int, float: Valid year or NaN if invalid.
    """
    
    try:
        if pd.isna(value):
            return np.nan  # Handle missing values

        # Handle float values (e.g., 2021.0 -> 2021)
        if isinstance(value, float):
            value = int(value)

        # Ensure it's a string and remove whitespace
        value = str(value).strip()

        # Remove leading zeros (e.g., "02019" -> "2019")
        if value.isdigit():
            value = str(int(value))
        else:
            return np.nan  # Invalid format

        # If value is two-digit, assume it's from the 2000s (e.g., "21" -> "2021")
        if len(value) == 2 and int(value) <= int(str(current_year)[-2:]):
            value = "20" + value

        year = int(value)

        # Check for reasonable year values
        if 1900 <= year <= current_year:
            return year
        else:
            return np.nan  # Invalid years

    except Exception as e:
        logger.error(f"Error in clean_built_year: {e}, Value: {value}")
        return np.nan


# Function to calculate age possession
def calculate_age_possession(built_year: int | float) -> str:
    
    """
    Calculates the age possession category from the built year.

    Args:
        built_year (int, float): Year of construction.

    Returns:
        str: Age possession category.
    
    """
    
    try:
        if pd.isna(built_year):
            return "Undefined"

        age = current_year - int(built_year)

        if age < 1:
            return "Under Construction"
        elif 1 <= age <= 3:
            return "Relatively New"
        elif 4 <= age <= 10:
            return "New Property"
        elif 11 <= age <= 20:
            return "Moderately Old"
        else:
            return "Old Property"

    except Exception as e:
        logger.error(f"Error in calculate_age_possession: {e}, Year: {built_year}")
        return "Undefined"



def Area_convert_to_sqft(area) -> float:
    
    """
    Convert the Area column to square feet.

    Args:
        area (str): Input area value.

    Returns:
        float: Area in square feet.

    """

    # Conversion factors
    KANAL_TO_SQFT = 5445
    MARLA_TO_SQFT = 272.25
    SQYD_TO_SQFT = 9
    
    try:
        if pd.isna(area):  # Handle NaN values
            return np.nan
        area = str(area)  # Ensure it's a string
        if "Kanal" in area:
            num = float(re.findall(r"[\d.]+", area)[0])
            return round(num * KANAL_TO_SQFT)
        elif "Marla" in area:
            num = float(re.findall(r"[\d.]+", area)[0])
            return round(num * MARLA_TO_SQFT)
        elif "Sq. Yd." in area:
            num = float(re.findall(r"[\d.]+", area)[0])
            return round(num * SQYD_TO_SQFT)
        else:
            return None  # If format is unknown
    except Exception as e:
        logger.error(f"Error in Area_convert_to_sqft: {e}, Area: {area}")
        
    


def after_covert_area_format_number(value:float)->int:
    
    """
    
    Convert the Area column to integer.

    Args:
        value (float): Input area value.

    Returns:

        int: Area in integer.
    
    """
    
    try:
        if pd.isna(value):
            return value
        return int(round(value))  # Always round and convert to integer
    except Exception as e:
        logger.error(f"Error in after_covert_area_format_number: {e}, Value: {value}")
        



def make_column_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Create a new column for price per square foot.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new column.
    
    """
    
    try:
        df["price_per_sqft"] = round(df["price"] * 10000000 / df["area"],2)
        return df
    except Exception as e:
        logger.error(f"Error in make_column_price_per_sqft: {e}")
        


def extract_colony_province(sub_name:object):
    
    """
    Extract colony and province from the sub_name column.

    Args:
        sub_name (object): Input sub_name value.

    Returns:

            tuple: Colony and province values.
    
    """
    
    try:
        parts = sub_name.split(",")  # Assuming values are separated by commas
        colony = parts[0].strip() if len(parts) > 0 else None
        province = parts[-1].strip() if len(parts) > 1 else None
        return colony, province
    except Exception as e:
        logger.error(f"Error in extract_colony_province: {e}")



def preprocess_colony_and_Location(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Extract colony and province from the sub_name column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new columns.
    
    """
    
    try:
        df['colony']=df['colony'].str.split('-').str.get(0)
        df['colony']=df['colony'].str.replace('Al','Al Ahmad')
        df['society']=df['Location'].str.split(',').str.get(0)  # make a society column
        return df 
    except Exception as e:
        logger.error(f"Error in preprocess_colony_and_Location: {e}")



def House_type(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Extract type from the House type and rename the House type to property type

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new columns.
    
    """
    try:
        df['House Type']=df['House Type'].str.split().str.get(1)
        df.rename(columns={"House Type":'property Type'},inplace=True)
        return df 
    except Exception as e:
        logger.error(f"Error in House_type(): {e}")










def main():
    try:
        # Load data
        df=pd.read_csv('data/raw/data_version_1.csv')    


    #    # Clean the price column
        df['price'] = df['price'].apply(lambda x: clean_price(df, x.split(' ')) if isinstance(x, str) else x)

        # Apply cleaning and transformation
        df['Year'] = df['Year'].apply(clean_built_year)
        df['Age Possession'] = df['Year'].apply(calculate_age_possession)

        # Convert area to square feet
        df["area"] = df["Area"].apply(Area_convert_to_sqft)

        # # Convert area to integer
        df["area"] = df["area"].apply(after_covert_area_format_number)
        #make price per sqft column
        df=make_column_price_per_sqft(df)
        # Extract colony and province
        df["colony"], df["province"] = zip(*df["sub_name"].apply(extract_colony_province))
       
       # Preprocess colony and Location
        df=preprocess_colony_and_Location(df)

       

        # Drop unnecessary columns
        df=drop_columns(df, ['House Number', 'Name', 'sub_name', 'description','Area','Year'])

        # House function call
        df=House_type(df)
        # Save the preprocessed data
        df.to_csv('data/processed/data_version_preprocessed.csv', index=False)
        
        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == '__main__':
    main()





