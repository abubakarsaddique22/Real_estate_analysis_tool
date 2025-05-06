import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/imputed_data.csv")
    df = df.drop(columns=[
        'price_per_sqft', 'area_room_ratio', 'Purpose', 'Location', 'province', 'society'
    ])
    return df

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

# Get input from session state
if "user_input" not in st.session_state:
    st.error("Please go back and enter your preferences.")
    st.stop()

user_input = st.session_state.user_input

st.title("📢 Recommendations")

recommendations, filters_used = content_based_recommend(df, user_input)

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
