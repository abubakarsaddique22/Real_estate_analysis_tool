# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Real Estate Price Prediction")

# st.write("# Welcome to the Real Estate Price Prediction App! 👋")

# # Load Data
# df = pd.read_csv('data/processed/imputed_data.csv')

# # Load Model
# try:
#     with open('models/model.pkl', 'rb') as file:
#         model = pickle.load(file)
# except FileNotFoundError:
#     st.error("Model file not found. Ensure that 'models/model.pkl' exists.")
#     st.stop()

# # Load Preprocessor
# try:
#     with open('models/preprocessor.pkl', 'rb') as f:
#         preprocessor = pickle.load(f)  # Assuming it's a ColumnTransformer
# except FileNotFoundError:
#     st.error("Preprocessor file not found. Ensure that 'models/preprocessor.pkl' exists.")
#     st.stop()

# st.header('Enter Your Property Details')

# # User Inputs
# property_type = st.selectbox('Property Type', ['Flats', 'Houses'])
# colony = st.selectbox('Colony', sorted(df['colony'].unique().tolist()))
# bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['Bedrooms'].unique().tolist())))
# bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['Bathrooms'].unique().tolist())))
# property_age = st.selectbox('Property Age', sorted(df['Age Possession'].unique().tolist()))
# City = st.selectbox('City', sorted(df['City'].unique().tolist()))
# province = st.selectbox('Province', sorted(df['province'].unique().tolist()))
# Kitchens = st.selectbox('Kitchens', sorted(df['Kitchens'].unique().tolist()))
# Servant_Quarters = st.selectbox('Servant Quarters', sorted(df['Servant Quarters'].unique().tolist()))
# Store_Rooms = st.selectbox('Store Rooms', sorted(df['Store Rooms'].unique().tolist()))
# Parking_Spaces = st.selectbox('Parking Spaces', sorted(df['Parking Spaces'].unique().tolist()))
# built_up_area = float(st.number_input('Built Up Area (sqft)'))

# # Define columns
# input_columns = [
#     'property Type', 'City', 'colony', 'Bedrooms', 'Bathrooms',
#     'Age Possession', 'area', 'Servant Quarters', 'Store Rooms',
#     'Kitchens', 'Parking Spaces', 'province'
# ]

# if st.button('Predict'):
#     # Form a DataFrame
#     one_df = pd.DataFrame([[property_type, City, colony, bedrooms, bathroom,
#                             property_age, built_up_area, Servant_Quarters, Store_Rooms, 
#                             Kitchens, Parking_Spaces, province]], columns=input_columns)
    
#     st.write("### Checking Input DataFrame Before Transformation:")
#     st.write(one_df)

#     # Get expected feature names from the preprocessor
#     try:
#         expected_columns = preprocessor.get_feature_names_out()  # Only works if ColumnTransformer is fitted
#     except AttributeError:
#         expected_columns = input_columns  # If get_feature_names_out doesn't exist, assume input columns

#     # Ensure column names match
#     if list(one_df.columns) != list(expected_columns):
#         st.error(f"Column mismatch! Expected {expected_columns}, but got {list(one_df.columns)}")
#         st.stop()

#     # Apply preprocessing using the preprocessor pipeline
#     try:
#         transformed_data = preprocessor.transform(one_df)
#     except Exception as e:
#         st.error(f"Error while preprocessing input: {e}")
#         st.stop()

#     # Predict
#     base_price = np.expm1(model.predict(transformed_data))[0]
#     low = base_price - 0.22
#     high = base_price + 0.22

#     # Display
#     st.text(f"The price of the property is between {round(low, 2)} Cr and {round(high, 2)} Cr")

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Real Estate Price Prediction")

st.write("# Welcome to the Real Estate Price Prediction App! 👋")

# Load Data (ensure this path is correct)
df = pd.read_csv('data/processed/imputed_data.csv')

# Load Model
try:
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Ensure that 'models/model.pkl' exists.")
    st.stop()

# Load Preprocessor
try:
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)  # Assuming it's a ColumnTransformer
except FileNotFoundError:
    st.error("Preprocessor file not found. Ensure that 'models/preprocessor.pkl' exists.")
    st.stop()

st.header('Enter Your Property Details')

# User Inputs
property_type = st.selectbox('Property Type', ['Flats', 'Houses'])
colony = st.selectbox('Colony', sorted(df['colony'].unique().tolist()))
bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['Bedrooms'].unique().tolist())))
bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['Bathrooms'].unique().tolist())))
property_age = st.selectbox('Property Age', sorted(df['Age Possession'].unique().tolist()))
City = st.selectbox('City', sorted(df['City'].unique().tolist()))
province = st.selectbox('Province', sorted(df['province'].unique().tolist()))
Kitchens = st.selectbox('Kitchens', sorted(df['Kitchens'].unique().tolist()))
Servant_Quarters = st.selectbox('Servant Quarters', sorted(df['Servant Quarters'].unique().tolist()))
Store_Rooms = st.selectbox('Store Rooms', sorted(df['Store Rooms'].unique().tolist()))
Parking_Spaces = st.selectbox('Parking Spaces', sorted(df['Parking Spaces'].unique().tolist()))
built_up_area = float(st.number_input('Built Up Area (sqft)'))

# Define input columns
input_columns = [
    'property Type', 'City', 'colony', 'Bedrooms', 'Bathrooms',
    'Age Possession', 'area', 'Servant Quarters', 'Store Rooms',
    'Kitchens', 'Parking Spaces', 'province'
]

if st.button('Predict'):
    # Prepare input data
    input_data = {
        'property Type': property_type,
        'City': City,
        'colony': colony,
        'Bedrooms': bedrooms,
        'Bathrooms': bathroom,
        'Age Possession': property_age,
        'area': built_up_area,
        'Servant Quarters': Servant_Quarters,
        'Store Rooms': Store_Rooms,
        'Kitchens': Kitchens,
        'Parking Spaces': Parking_Spaces,
        'province': province
    }

    # Create DataFrame
    one_df = pd.DataFrame([input_data])

    # Ensure columns match with expected ones from preprocessor
    try:
        expected_columns = preprocessor.get_feature_names_out()
    except AttributeError:
        # If preprocessor does not have get_feature_names_out() method, use the input columns
        expected_columns = input_columns

    # Reorder the columns to match expected ones
    one_df = one_df[expected_columns]

    # Apply preprocessing
    try:
        transformed_data = preprocessor.transform(one_df)
    except Exception as e:
        st.error(f"Error while preprocessing input: {e}")
        st.stop()

    # Prediction
    try:
        base_price = np.expm1(model.predict(transformed_data))[0]  # Inverse transform for price
        low = base_price - 0.22
        high = base_price + 0.22
        st.text(f"The price of the property is between {round(low, 2)} Cr and {round(high, 2)} Cr")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

