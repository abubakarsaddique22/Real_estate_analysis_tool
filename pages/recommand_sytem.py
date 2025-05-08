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

# Content-based recommendation function
def content_based_recommend(df, user_input, top_n=5):
    filters = ['City', 'property Type', 'Age Possession', 'colony']
    df_filtered = df.copy()

    for i in range(len(filters) + 1):
        temp_df = df_filtered.copy()
        for col in filters[:len(filters) - i]:
            temp_df = temp_df[temp_df[col].str.lower() == user_input[col].lower()]

        if not temp_df.empty:
            temp_df['score'] = (
                (temp_df['price'] - user_input['price']).abs() +
                (temp_df['area'] - user_input['area']).abs()
            )
            return temp_df.sort_values('score').head(top_n).drop(columns='score'), len(filters) - i

    return pd.DataFrame(), 0

# Load data
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

    # User input fields
    city = st.selectbox("City", city_options)
    property_type = st.selectbox("Property Type", type_options)
    age_possession = st.selectbox("Age Possession", age_options)
    colony = st.selectbox("Colony", colony_options)
    price = st.number_input("Target Price (Millions)", min_value=0.1, value=3.0, step=0.1)
    area = st.number_input("Target Area (sqft)", min_value=100.0, value=1500.0, step=50.0)

    submit = st.form_submit_button("🔍 Recommend Properties")

# Handle form submission
if submit:
    user_input = {
        'City': city,
        'property Type': property_type,
        'Age Possession': age_possession,
        'colony': colony,
        'price': price,
        'area': area
    }

    # Get recommendations based on the user input
    recommendations, filters_used = content_based_recommend(df, user_input)

    # Display the results
    if not recommendations.empty:
        if filters_used < 4:
            st.info(f"⚠️ No exact match found — relaxed filters to show similar listings (used {filters_used} filters).")
        
        st.dataframe(recommendations[[  # Display relevant columns
            'City', 'property Type', 'price', 'area', 'Bedrooms', 'Bathrooms',
            'Parking Spaces', 'Kitchens', 'Store Rooms', 'Servant Quarters',
            'Age Possession', 'colony'
        ]].reset_index(drop=True))
    else:
        st.error("❌ Sorry! No properties found. Try changing your filters.")
