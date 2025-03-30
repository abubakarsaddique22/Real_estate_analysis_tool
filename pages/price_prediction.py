import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")

st.write("# Welcome to page 1! 👋")

with open('df.pkl','rb') as file:
    df = pickle.load(file)

with open('models/model.pkl','rb') as file:
    model = pickle.load(file)

# df.drop(columns=['Unnamed: 0'],inplace=True)
# st.write(df.columns)
# st.write(df.head())
st.header('Enter your inputs')

# property_type
property_type = st.selectbox('Property Type',['Flats','Houses'])

# colony
colony = st.selectbox('colony',sorted(df['colony'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedrooms',sorted(df['Bedrooms'].unique().tolist())))
# 
bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['Bathrooms'].unique().tolist())))

# # balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['Age Possession'].unique().tolist()))
City= st.selectbox('City',sorted(df['City'].unique().tolist()))
province = st.selectbox('province',sorted(df['province'].unique().tolist()))


Kitchens= st.selectbox('Kitchens',sorted(df['Kitchens'].unique().tolist()))
Servant_Quarters= st.selectbox('Servant Quarters',sorted(df['Servant Quarters'].unique().tolist()))
Store_Rooms= st.selectbox('Store Rooms',sorted(df['Store Rooms'].unique().tolist()))
Parking_Spaces= st.selectbox('Parking Spaces',sorted(df['Parking Spaces'].unique().tolist()))
built_up_area = float(st.number_input('Built Up Area'))

#  'City', 'property Type', 'Parking Spaces', 'Bedrooms',
#        'Bathrooms', 'Servant Quarters', 'Kitchens', 'Store Rooms', 'price',
#        'Age Possession', 'area', 'colony', 'province']

if st.button('Predict'):

# #     # form a dataframe
     data = [[property_type,City, colony, bedrooms, bathroom, property_age, built_up_area, Servant_Quarters, Store_Rooms,Kitchens
              , Parking_Spaces, province]]
     columns = ['property Type', 'City','colony', 'Bedrooms', 'Bathrooms',
                'Age Possession', 'area', 'Servant Quarters', 'Store Rooms',
                'Kitchens','Parking Spaces', 'province']
# City,property Type,Parking Spaces,Bedrooms,Bathrooms,Servant Quarters,Kitchens,Store Rooms,price,Age Possession,area,colony,province
     # Convert to DataFrame
     one_df = pd.DataFrame(data, columns=columns)

     st.dataframe(one_df)

#     # predict
     base_price = np.expm1(model.predict(one_df))[0]
     low = base_price - 0.22
     high = base_price + 0.22

# #     # display
     st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))