import streamlit as st

# Set page config
st.set_page_config(page_title="Smart Property Decisions with AI", layout="wide")

# Optional: Global background color
page_bg = """
<style>
body {
    background-color: #f5f9ff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Hero Section
st.markdown(
    """
    <div style="background: linear-gradient(to right, #6a11cb, #2575fc); padding: 165px 5px; border-radius: 10px; text-align: center; color: white;">
        <h1 style="font-size: 3em;">Make Smart Property Decisions with AI</h1>
        <p style="font-size: 1.5em;">Analyze trends. Predict prices. Discover opportunities.</p>
        <a href="#" style="background-color: white; color: #2575fc; padding: 10px 30px; border-radius: 8px; text-decoration: none; font-weight: bold;">Get Started</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Features Section with background and border
st.markdown("---")
st.markdown("## 🔍 Features")

features_html = """
<style>
.feature-box {
    background:#2c3e50;
    border: 2px solid #d0e0ff;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    height: 300px;
    transition: 0.3s;
}
.feature-box:hover {
    background: #e0ecff;
    color: #000000;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
}
</style>

<div class="row" style="display: flex; gap: 20px; justify-content: center;">
    <div class="feature-box" style="flex: 1;">
        <h3>📈 Analytics</h3>
        <p>Explore colony-wise pricing, configurations, and trends.</p>
    </div>
    <div class="feature-box" style="flex: 1;">
        <h3>💸 Price Prediction</h3>
        <p>AI-powered price estimation in seconds.</p>
    </div>
    <div class="feature-box" style="flex: 1;">
        <h3>🧠 AI Recommendations</h3>
        <p>Personalized suggestions based on your inputs.</p>
    </div>
</div>
"""

st.markdown(features_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------------

# Full Feature Cards
st.markdown("---")
st.markdown("## ✅ what do ")

features_html = """
<style>
.feature-box_one {
    background: #2c3e50; /* Dark card background */
    color: #f0f0f0;       /* Light text color */
    border: 1px solid #4a6fa5;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    height: 450px;
    transition: 0.3s;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.05);
}
.feature-box_one:hover {
    background: #e0ecff;
    color: #000000;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
}
</style>

<div class="row" style="display: flex; gap: 20px; justify-content: center;">
    <div class="feature-box_one" style="flex: 1;">
        <h2>📈 Analytics </h2>
        <h3>Explore Real Estate Trends Like a Pro</h3>
        <p>Access interactive visualizations that reveal colony-wise pricing, room configurations, and market movement. With filters and dynamic charts, you can make informed decisions without being a data scientist.</p>
    </div>
    <div class="feature-box_one" style="flex: 1;">
       <h2>💸 Price Prediction</h2>
        <h3>Instant AI-Powered Valuation</h3>
        <p>Simply enter property features like bedrooms, area, and location  and get an accurate price estimate in seconds. Our ML model is trained on real data to deliver fast and reliable results.</p>
    </div>
    <div class="feature-box_one" style="flex: 1;">
        <h2>🧠 AI Recommendations</h2>
        <h3>Smart Suggestions Just for You</h3>
        <p>Let our AI recommend properties that match your preferences. Using intelligent filtering based on your inputs, you'll discover the best-fitting listings effortlessly saving time and effort.</p>
    </div>
</div>
"""

st.markdown(features_html, unsafe_allow_html=True)




#----------------------------------------------------------------------------------------
# How It Works Section
st.markdown("---")
st.markdown("## ✅ How It Works")

features_html = """
<style>
.feature-box_two {
    background: #2c3e50; /* Dark card background */
    color: #f0f0f0;       /* Light text color */
    border: 1px solid #4a6fa5;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    height: 380px;
    transition: 0.3s;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.05);
}
.feature-box_two:hover {
    background: #e0ecff;
    color: #000000;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
}
</style>

<div class="row" style="display: flex; gap: 20px; justify-content: center;">
    <div class="feature-box_two" style="flex: 1;">
        <h3>📝 Enter Property Info</h3>
        <p>Select colony, area, number of bedrooms, bathrooms, and other features. The more accurate your input, the better the predictions..</p>
    </div>
    <div class="feature-box_two" style="flex: 1;">
        <h3>📊 Predict Prices & View Trends</h3>
        <p>Instantly get AI-generated property valuation plus detailed colony-level analytics and historical price trends.</p>
    </div>
    <div class="feature-box_two" style="flex: 1;">
        <h3>🎯 Discover Best-Match Properties</h3>
        <p>Get intelligent recommendations tailored to your input and preferences—save time and find ideal options quickly.</p>
    </div>
</div>
"""


st.markdown(features_html, unsafe_allow_html=True)


# -----------------------------------------------------------
footer_html = """
<style>
.footer {
    # background-color: #2c3e50;
    color: #f0f0f0;
    padding-top:30px;
    border-top: 1px solid #4a6fa5;
    margin-top: 40px;
    font-size: 16px;
}
.footer h3 {
    margin-top: 0;
    color: #ffffff;
}
.footer a {
    color: #f0f0f0;
    text-decoration: none;
}
.footer a:hover {
    color: #ffffff;
    text-decoration: underline;
}
.footer .footer-columns {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 30px;
}
.footer .column {
    flex: 1;
    min-width: 200px;
}
.footer-bottom {
    text-align: center;
    margin-top: 30px;
    color: #aaa;
    font-size: 14px;
}
</style>

<div class="footer">
    <div class="footer-columns">
        <div class="column">
            <h3>🏡 RealEstate Insights</h3>
            <p>Empowering smarter property decisions with AI-driven price predictions, analytics, and personalized recommendations.</p>
        </div>
        <div class="column">
            <h3>Quick Links</h3>
            <p><a href="#">Home</a></p>
            <p><a href="#">Price Prediction</a></p>
            <p><a href="#">Analytics</a></p>
            <p><a href="#">Recommendations</a></p>
        </div>
        <div class="column">
            <h3>Contact</h3>
            <p>Email: support@realestateinsights.ai</p>
            <p>Phone: +92-300-1234567</p>
            <p>Lahore, Pakistan</p>
        </div>
    </div>
    <div class="footer-bottom">
        © 2025 RealEstate Insights. All rights reserved.
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)