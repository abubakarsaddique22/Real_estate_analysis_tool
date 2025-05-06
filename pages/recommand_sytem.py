import streamlit as st
import pandas as pd

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/imputed_data.csv")
    df = df.drop(columns=[
        'price_per_sqft', 'area_room_ratio', 'Purpose', 'Location', 'province', 'society'
    ])
    return df

df = load_data()

# Dropdown options
city_options = df['City'].dropna().unique().tolist()
type_options = df['property Type'].dropna().unique().tolist()
age_options = df['Age Possession'].dropna().unique().tolist()
colony_options = df['colony'].dropna().unique().tolist()

# Title and input form
st.title("🏠 Real Estate Recommendation System")

with st.form("recommend_form"):
    st.subheader("📋 Enter Your Preferences")

    city = st.selectbox("City", city_options)
    property_type = st.selectbox("Property Type", type_options)
    age_possession = st.selectbox("Age Possession", age_options)
    colony = st.selectbox("Colony", colony_options)
    price = st.number_input("Target Price (Millions)", min_value=0.1, value=3.0, step=0.1)
    area = st.number_input("Target Area (sqft)", min_value=100.0, value=1500.0, step=50.0)

    submit = st.form_submit_button("🔍 Recommend Properties")

# Save form input and switch to result page
if submit:
    st.session_state.user_input = {
        'City': city,
        'property Type': property_type,
        'Age Possession': age_possession,
        'colony': colony,
        'price': price,
        'area': area
    }
    st.switch_page("pages/2_Recommender_Result.py")
