import streamlit  as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns 


st.set_page_config(page_title="Plotting Demo")

st.title('Analytics')

# new_df = pd.read_csv('datasets/data_viz1.csv')
# feature_text = pickle.load(open('datasets/feature_text.pkl','rb'))

# # Load Preprocessor
try:
    with open('models/preprocessor.pkl', 'rb') as f:
        data = pickle.load(f)  # Assuming it's a ColumnTransformer
except FileNotFoundError:
    st.error("Preprocessor file not found. Ensure that 'models/preprocessor.pkl' exists.")
    st.stop()





# feature importance according to price

# Select numeric features for correlation calculation
numeric_cols = data.select_dtypes(include='number')

# Calculate correlation with 'price'
correlations = numeric_cols.corr()['price'].drop('price').abs().sort_values(ascending=False)

# Convert correlations to a dictionary (for the WordCloud)
correlation_dict = correlations.to_dict()

# Create a Word Cloud
plt.figure(figsize=(12, 8))
wordcloud = WordCloud(width=800, height=600, background_color='white', colormap='viridis')
wordcloud.generate_from_frequencies(correlation_dict)


# Display the Word Cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Feature Importance for Property Price')
st.pyplot(plt)  # ✅ Use this instead of plt.show()






# area vs price according to property tyep
# 1. in this case if user select house then first graph show
# 2. in this case if user select flats then second graph show

st.header('Area Vs Price')

property_type = st.selectbox('Select Property Type', ['flat','house'],key="feature_selectbox_1")

if property_type == 'house':
    fig1 = px.scatter(data[data['property_type'] == 'Houses'], x="area", y="price", color="Bedrooms", title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)
else:
    fig1 = px.scatter(data[data['property_type'] == 'Flats'], x="area", y="price", color="Bedrooms", title="Area Vs Price")

    st.plotly_chart(fig1, use_container_width=True)





# bedrooms and kitchen according to colony
st.header('Bedrooms Pie Chart')
select_colony = st.selectbox('Select colony',['overall','specific colony'],key="feature_selectbox_2")
if select_colony == 'overall':
    fig2 = px.pie(data, names='Bedrooms',title='Bedrooms vs colony')

    st.plotly_chart(fig2, use_container_width=True)
else:
    specific_colony=st.selectbox('select specific',sorted(data['colony'].unique().to_list()))
    fig2 = px.pie(data[data['colony'] ==specific_colony], names='Bedrooms',title="Area Vs Price")

    st.plotly_chart(fig2, use_container_width=True)


st.header('Kitchens Pie Chart')
select_kitchen = st.selectbox('Select colony',['overall','specific colony'],key="feature_selectbox_3")

if select_kitchen == 'overall':
    fig2 = px.pie(data, names='Kitchens')

    st.plotly_chart(fig2, use_container_width=True)
else:
    specific_colony=st.selectbox('select specific',sorted(data['colony'].unique().to_list()))
    fig2 = px.pie(data[data['colony'] ==specific_colony], names='Kitchens')

    st.plotly_chart(fig2, use_container_width=True)





# average price by bedrooms and kitchens 
st.header('Average Price By BK')
select_BK = st.selectbox('Select Bk',['Bedrooms','Kitchens'],key="feature_selectbox_4")

if select_BK=='Bedrooms':
        # Calculate average price by bedrooms
    avg_price_by_bedrooms = data.groupby('Bedrooms')['price'].mean().reset_index()
    avg_price_by_bedrooms = avg_price_by_bedrooms[avg_price_by_bedrooms['Bedrooms'] >= 1]

    # Helper function to format price (Cr or Lakhs)
    def format_price(value):
        if value >= 1:
            return f"{value:.2f} Cr"  # Values ≥ 1 Cr
        else:
            return f"{value * 100:.0f} L"  # Values < 1 Cr (converted to Lakhs)

    # Apply formatting to display Cr/L
    avg_price_by_bedrooms['formatted_price'] = avg_price_by_bedrooms['price'].apply(format_price)

    # Bar Chart - Bedrooms vs. Average Price (with formatted hover)
    fig = px.bar(avg_price_by_bedrooms,
                x='Bedrooms',
                y='price',
                title='Average Price by Number of Bedrooms',
                labels={'price': 'Average Price (Cr/L)', 'Bedrooms': 'Number of Bedrooms'},
                color='price',
                color_continuous_scale='Viridis',
                hover_data=['formatted_price'])

    # # Show chart
    st.plotly_chart(fig, use_container_width=True)
else:
    # Calculate average price by number of kitchens
    avg_price_by_kitchens = data.groupby('Kitchens')['price'].mean().reset_index()

    # Filter to exclude 0 kitchens and include only 1 or more
    avg_price_by_kitchens = avg_price_by_kitchens[avg_price_by_kitchens['Kitchens'] >= 1]

    # Helper function to format price (Cr or Lakhs)
    def format_price(value):
        if value >= 1:
            return f"{value:.2f} Cr"  # Values ≥ 1 Cr
        else:
            return f"{value * 100:.0f} L"  # Values < 1 Cr (converted to Lakhs)

    # Apply formatting to display Cr/L
    avg_price_by_kitchens['formatted_price'] = avg_price_by_kitchens['price'].apply(format_price)

    # Bar Chart - Kitchens vs. Average Price (with formatted hover)
    fig = px.bar(avg_price_by_kitchens,
                x='Kitchens',
                y='price',
                title='Average Price by Number of Kitchens (1 and Above)',
                labels={'price': 'Average Price (Cr/L)', 'Kitchens': 'Number of Kitchens'},
                color='price',
                color_continuous_scale='Viridis',
                hover_data=['formatted_price'])

    # Show chart
    st.plotly_chart(fig, use_container_width=True)





# avg price per colony
st.header('average price per colony')

# Calculate avg price per colony
avg_price_per_colony = data.groupby('colony')['price'].mean().reset_index()
avg_price_per_colony.rename(columns={'price': 'avg_price_per_colony'}, inplace=True)

# Merge with original DataFrame
data = pd.merge(data, avg_price_per_colony, on='colony', how='left')


# ✅ Convert to strings
data['City'] = data['City'].astype(str)
data['colony'] = data['colony'].astype(str)


# Treemap: Property Distribution by Colony with Avg Price on Hover
fig = px.treemap(data, path=['City', 'colony'],
                 title='Property Distribution by Colony',
                 color='City',
                 hover_data=['avg_price_per_colony'])

fig.update_layout(
    height=800,  # 🔺 Increase vertical size
    margin=dict(t=50, l=25, r=25, b=25)
)

st.plotly_chart(fig, use_container_width=True)




st.header('Side by Side Distplot for Property Type')



# Create the figure
fig3 = plt.figure(figsize=(10, 4))
house_prices = data[data['property_type'] == 'Houses']['price']
flat_prices = data[data['property_type'] == 'Flats']['price']
# Use sns.distplot() with clean data
sns.histplot(house_prices, label='House',  kde=True, color='skyblue')
sns.histplot(flat_prices, label='Flat', kde=True, color='salmon')

plt.legend()
plt.title("Price Distribution: House vs Flat")
plt.xlabel("Price")
plt.ylabel("Density")

# Show plot in Streamlit
st.pyplot(fig3)
