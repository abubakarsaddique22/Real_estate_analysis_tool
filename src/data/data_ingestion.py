import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import re 

import logging

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


def load_data(file_path:str)->pd.DataFrame:
    """
    Read data from file

    """
    try:
        df= pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        return df
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """
    Preprocess the data

    """
    try:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df.rename(columns={'Rawalpindi Houses':'City','Price':'sub_name','Type':"description"}, inplace=True)

        # For House Number column
        df['House Number']=df['House Number'].str.split(" ").str.get(-1)
        # check duplicated value in House Number column
        df.duplicated(subset='House Number').sum()
        df[df['House Number'].duplicated()]

        # remove the duplicated value
        df.drop_duplicates(subset='House Number',keep='first', inplace=True)

        # For city column
        def separate_category(value):
            if pd.isna(value):
                return  None
            cat_match = re.search(r"([A-Za-z ]+)", value)   # Extracts category/text values
            cat_value = cat_match.group(1).strip() if cat_match else None

            return cat_value

        # Apply the function to split House Type into numerical and categorical columns
        df["House Type"] = df['City'].apply(lambda x: pd.Series(separate_category(x)))
        df['City']=df['City'].str.split(' ').str.get(0)

        # Main Features column
        """
        make some columns like year,Parking Spaces
        
        """

        # year columns
        df['Year']=df['Main Features'].str.split('|').str.get(0)
        df['Year']=df['Year'].str.split(':').str.get(1)

        # parking spaces
        df['Parking Spaces']=df['Main Features'].str.split('|').str.get(1).str.split(':').str.get(1)
        df['Parking Spaces'].unique()
        df.drop(['Main Features'], axis=1, inplace=True)


        #  Rooms column
        """
        In Rooms column we create multiply new columns 
        like Bedrooms,Bathrooms,Servant Quarters,Kitchens,Stores Rooms
        
        """


        # Function to extract specific room details
        def extract_room_info(room_str, keyword):
            if pd.isna(room_str):
                return None
            match = re.search(rf"{keyword}:?\s*(\d+)", room_str, re.IGNORECASE)
            return int(match.group(1)) if match else 0

        # Extract relevant information
        df['Bedrooms'] = df['Rooms'].apply(lambda x: extract_room_info(x, 'Bedrooms'))
        df['Bathrooms'] = df['Rooms'].apply(lambda x: extract_room_info(x, 'Bathrooms'))
        df['Servant Quarters'] = df['Rooms'].apply(lambda x: extract_room_info(x, 'Servant Quarters'))
        df['Kitchens'] = df['Rooms'].apply(lambda x: extract_room_info(x, 'Kitchens'))
        df['Store Rooms'] = df['Rooms'].apply(lambda x: extract_room_info(x, 'Store Rooms'))

        # Display the rows using sample and verify the data
        df[['Bedrooms', 'Bathrooms', 'Servant Quarters', 'Kitchens','Store Rooms']].sample(10)
        # Drop the original column
        df.drop(['Rooms'], axis=1, inplace=True)


        # Details columns
        """ 
        Here is Details columns in make new columns such as 
        Area,price,Purpose,Location
        
        """
        df['price']=df['Details'].str.split(',').str.get(1).str.split(':').str.get(1)
        df['Area']=df['Details'].str.split(',').str.get(6).str.split(':').str.get(1)
        df['Purpose']=df['Details'].str.split(',').str.get(7).str.split(':').str.get(1)

        def extract_detail(detail_str, keyword):
            if pd.isna(detail_str):
                return None
            match = re.search(rf"{keyword}:?\s*([A-Za-z0-9,. ]+)", detail_str, re.IGNORECASE)
            return match.group(1).strip() if match else None

        # Extract relevant details

        df['Location'] = df['Details'].apply(lambda x: extract_detail(x, 'Location'))
        df['Location']=df['Location'].str.replace('Bath','')


        # Note:--> I will work further on location column

        # Drop the original column
        df.drop(['Details'], axis=1, inplace=True)

        # For price column
        df['price']=df['price'].str.replace("PKR",'')
        
        df['price'].str.split(' ').str.get(2)
        logging.info("Data Preprocessed successfully")
        return df 
    except Exception as e:
        logging.error("Error Preprocessing data: %s", e)
        raise


def save_data(df:pd.DataFrame,data_path:str)->pd.DataFrame:
    """
    Save the data to file
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, "data_version_1.csv"), index=False)
        logging.info("Data saved successfully")
    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise


def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        # random_state = params['data_ingestion']['random_state']

        df=load_data(file_path='data/raw/final_data_unclean.csv')
        final_df=preprocess_data(df)
        # train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=random_state)
        save_data(df,data_path='data')
        logging.info("Data Ingestion completed")
    except Exception as e:
        logging.error("Error in data ingestion: %s", e)
        raise

if __name__ == "__main__":
    main()
