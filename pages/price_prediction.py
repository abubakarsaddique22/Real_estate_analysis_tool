import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Real Estate Price Prediction")

st.write("# Welcome to the Real Estate Price Prediction App! 👋")

# Load Data
# df = pd.read_csv('data/processed/imputed_data.csv')

# Load Model
try:
    with open('models/model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Ensure that 'models/model.pkl' exists.")
    st.stop()

# # Load Preprocessor
try:
    with open('models/final_data.pkl', 'rb') as f:
        df = pickle.load(f)  # Assuming it's a ColumnTransformer
except FileNotFoundError:
    st.error("Preprocessor file not found. Ensure that 'models/preprocessor.pkl' exists.")
    st.stop()

st.header('Enter Your Property Details')

# st.dataframe(df)
# User Inputs
# with st.form("recommend_form"):
City = st.selectbox('City', sorted(df['City'].unique().tolist()))
property_type = st.selectbox('Property Type', ['Flats', 'Houses'])
parking_spaces = st.selectbox('Parking Spaces', sorted(df['parking_spaces'].unique().tolist()))
Bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['Bedrooms'].unique().tolist())))
Bathrooms = float(st.selectbox('Number of Bathrooms', sorted(df['Bathrooms'].unique().tolist())))
servant_Quarters = st.selectbox('Servant Quarters', sorted(df['servant_Quarters'].unique().tolist()))
Kitchens = st.selectbox('Kitchens', sorted(df['Kitchens'].unique().tolist()))
store_rooms = st.selectbox('Store Rooms', sorted(df['store_rooms'].unique().tolist()))
age_possession = st.selectbox('Property Age', sorted(df['age_possession'].unique().tolist()))
area = float(st.number_input('Built Up Area (sqft)'))
colony = st.selectbox('Colony', sorted(df['colony'].unique().tolist()))
province = st.selectbox('Province', sorted(df['province'].unique().tolist()))
# submit = st.form_submit_button("click predict button ⬇️")

if st.button('Predict'):

    data=[[City,property_type,parking_spaces,Bedrooms,Bathrooms,
        servant_Quarters, Kitchens, store_rooms,
        age_possession, area, colony, province]]

    columns = ['City', 'property_type', 'parking_spaces', 'Bedrooms', 'Bathrooms',
        'servant_Quarters', 'Kitchens', 'store_rooms',
        'age_possession', 'area', 'colony', 'province']

     # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #st.dataframe(one_df)

    # predict
    base_price = np.expm1(model.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))

    # # Display
    # st.text(f"The price of the property is between {round(low, 2)} Cr and {round(high, 2)} Cr")
