import os 
import pandas as pd
import numpy as np
from datetime import datetime
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

def impute_missing_property_type(row, median_price_medium_area) -> str:

    """
        Fills missing "Property Type" values based on area size:
        - Houses for large properties (≥ 2500 sq ft).
        - Flats for small properties (≤ 1500 sq ft).
        - Lower for medium properties (1500-2500 sq ft) with high price.
        - Upper for medium properties with lower price.
        - Defaults to "Houses" if area is missing.
    """
    
    try:
        if pd.isnull(row["property Type"]):
            if row["area"] >= 2500:
                return "Houses"
            elif row["area"] <= 1500:
                return "Flats"
            elif 1500 < row["area"] <= 2500:
                return "Lower" if row["price"] > median_price_medium_area else "Upper"
            else:
                return "Houses"  # Default to Houses if area is also missing
        return row["property Type"]
    except Exception as e:
        logger.error("Error in impute_missing_property_type: %s", e)
        raise

def impute_parking_spaces(df: pd.DataFrame) -> pd.DataFrame:

    """
        Imputes missing values in the 'Parking Spaces' column using the median value.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing parking spaces filled.
    """
    
    try:

        df['Parking Spaces'].fillna(df['Parking Spaces'].median(), inplace=True)
        return df
    except Exception as e:
        logger.error("Error in impute_parking_spaces: %s", e)
        raise

def impute_city(row, most_common_city) -> str:

    """
        Imputes missing 'City' values based on known city names in the row.
        If no match is found, it uses the most common city.

        Args:
            row (pd.Series): Input DataFrame row.
            most_common_city (str): Most frequent city for fallback.

        Returns:
            str: Imputed city value.
    """

    try:
        if pd.isnull(row["City"]):
            if "Lahore" in str(row.values):
                return "Lahore"
            elif "Rawalpindi" in str(row.values):
                return "Rawalpindi"
            elif "Karachi" in str(row.values):
                return "Karachi"
            else:
                return most_common_city  # Fill with most common city
        return row["City"]
    except Exception as e:
        logger.error("Error in impute_city: %s", e)
        raise

def main():

    """
        Main function to perform the following steps:
        - Load dataset
        - Impute missing values for 'Parking Spaces', 'Property Type', and 'City'
        - Save the updated DataFrame
    """

    try:
        # Load dataset
        df = pd.read_csv('data/processed/outlier_remove.csv')


        # Impute missing parking spaces
        df = impute_parking_spaces(df)

        # Compute median price for medium area (1500 - 2500 sq ft)
        median_price_medium_area = df[(df["area"] > 1500) & (df["area"] <= 2500)]["price"].median()

        # Apply property type imputation
        df["property Type"] = df.apply(lambda row: impute_missing_property_type(row, median_price_medium_area), axis=1)

        # Most frequent city (for fallback)
        most_common_city = df["City"].mode()[0]

        # Apply city imputation
        df["City"] = df.apply(lambda row: impute_city(row, most_common_city), axis=1)

        # Save the updated DataFrame
        df.to_csv('data/processed/imputed_data.csv', index=False)

        logger.info("Imputation process completed successfully.")

        # print(df["property Type"].value_counts())
        # print(df["City"].value_counts())
        # print(df.isnull().sum())

    except Exception as e:
        logger.error("Error occurred during main execution: %s", e)
        raise

if __name__ == '__main__':
    main()
