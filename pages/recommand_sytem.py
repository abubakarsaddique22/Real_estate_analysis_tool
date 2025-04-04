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

# UI
st.title("🏠 Real Estate Recommendation System")

st.sidebar.header("📋 Enter Your Preferences")

user_input = {
    'City': st.sidebar.selectbox("City", city_options),
    'property Type': st.sidebar.selectbox("Property Type", type_options),
    'Age Possession': st.sidebar.selectbox("Age Possession", age_options),
    'colony': st.sidebar.selectbox("Colony", colony_options),
    'price': st.sidebar.number_input("Target Price (Millions)", min_value=0.1, value=3.0, step=0.1),
    'area': st.sidebar.number_input("Target Area (sqft)", min_value=100.0, value=1500.0, step=50.0),
}

# Recommendation function with fallback logic
def content_based_recommend(df, user_input, top_n=5):
    filters = ['City', 'property Type', 'Age Possession', 'colony']
    df_filtered = df.copy()

    # Step-by-step fallback filtering
    for i in range(len(filters) + 1):  # 0 to 4
        temp_df = df_filtered.copy()
        for col in filters[:len(filters) - i]:  # Remove one filter each time
            temp_df = temp_df[temp_df[col].str.lower() == user_input[col].lower()]

        if not temp_df.empty:
            temp_df['score'] = (
                (temp_df['price'] - user_input['price']).abs() +
                (temp_df['area'] - user_input['area']).abs()
            )
            return temp_df.sort_values('score').head(top_n).drop(columns='score'), len(filters) - i

    return pd.DataFrame(), 0

# Only recommend when button is clicked
if st.button("🔍 Recommend Properties"):
    recommendations, filters_used = content_based_recommend(df, user_input)

    st.subheader("📢 Recommendations")
    if not recommendations.empty:
        if filters_used < 4:
            st.info(f"⚠️ No exact match found — relaxed filters to show similar listings (used {filters_used} filters).")

        st.dataframe(recommendations[[
            'City', 'property Type', 'price', 'area', 'Bedrooms', 'Bathrooms',
            'Parking Spaces', 'Kitchens', 'Store Rooms', 'Servant Quarters',
            'Age Possession', 'colony'
        ]].reset_index(drop=True))
    else:
        st.error("❌ Sorry! No properties found. Try changing your filters.")
