import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Whale Prediction App", layout="centered")

st.title("🎮 Mobile Game Whale Predictor")
st.markdown("""
Predict whether a player will convert to a paying customer (a 'whale') based on their early‑game behaviour.
""")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("models/whale_prediction_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'models/whale_prediction_model.pkl' exists.")
    st.stop()

# Sidebar with model info
st.sidebar.header("About")
st.sidebar.markdown("""
- **Model**: Logistic Regression (trained on 9,600 players)
- **Preprocessing**: Standard scaling + one‑hot encoding
- **Target**: `converted_to_payer` (1 = payer, 0 = non‑payer)
""")

# Input widgets
st.header("Enter Player Features")

# Numerical features (using number_input with min/max from dataset, you can adjust ranges)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=13, max_value=60, value=25, step=1)
    days_since_install = st.number_input("Days Since Install", min_value=1, max_value=90, value=10, step=1)
    sessions_last_7d = st.number_input("Sessions (last 7 days)", min_value=0, max_value=30, value=7, step=1)
    avg_session_length_min = st.number_input("Avg Session Length (minutes)", min_value=0.0, max_value=35.0, value=10.0, step=0.5)
    total_playtime_hours = st.number_input("Total Playtime (hours)", min_value=0.0, max_value=120.0, value=5.0, step=0.5)
    levels_completed = st.number_input("Levels Completed", min_value=0, max_value=60, value=15, step=1)

with col2:
    current_level = st.number_input("Current Level", min_value=1, max_value=60, value=15, step=1)
    tutorial_completed = st.selectbox("Tutorial Completed", [0, 1], format_func=lambda x: "Yes" if x else "No")
    num_friends_connected = st.number_input("Friends Connected", min_value=0, max_value=15, value=2, step=1)
    push_notifications_enabled = st.selectbox("Push Notifications Enabled", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ad_views = st.number_input("Ad Views", min_value=0, max_value=25, value=6, step=1)
    rewarded_ad_views = st.number_input("Rewarded Ad Views", min_value=0, max_value=15, value=2, step=1)
    store_visits = st.number_input("Store Visits", min_value=0, max_value=12, value=2, step=1)

col3, col4 = st.columns(2)

with col3:
    items_viewed_in_store = st.number_input("Items Viewed in Store", min_value=0, max_value=45, value=5, step=1)
    wishlist_items = st.number_input("Wishlist Items", min_value=0, max_value=10, value=1, step=1)
    days_active_last_30 = st.number_input("Days Active (last 30)", min_value=0, max_value=30, value=15, step=1)
    streak_days = st.number_input("Streak Days", min_value=0, max_value=50, value=4, step=1)

with col4:
    rage_quit_events = st.number_input("Rage Quit Events", min_value=0, max_value=15, value=3, step=1)
    level_fail_rate = st.number_input("Level Fail Rate", min_value=0.0, max_value=1.0, value=0.6, step=0.01, format="%.3f")
    social_shares = st.number_input("Social Shares", min_value=0, max_value=8, value=1, step=1)

# Categorical features
st.subheader("Categorical Attributes")
cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

with cat_col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with cat_col2:
    country = st.selectbox("Country", ["USA", "UK", "Canada", "Germany", "France", "Brazil", "Mexico", "India", "Philippines", "Indonesia", "Japan"])
with cat_col3:
    acquisition_channel = st.selectbox("Acquisition Channel", ["organic", "paid_social", "paid_search", "influencer", "referral"])
with cat_col4:
    device_type = st.selectbox("Device Type", ["Android", "iOS"])

# Collect all inputs into a DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "days_since_install": [days_since_install],
    "sessions_last_7d": [sessions_last_7d],
    "avg_session_length_min": [avg_session_length_min],
    "total_playtime_hours": [total_playtime_hours],
    "levels_completed": [levels_completed],
    "current_level": [current_level],
    "tutorial_completed": [tutorial_completed],
    "num_friends_connected": [num_friends_connected],
    "push_notifications_enabled": [push_notifications_enabled],
    "ad_views": [ad_views],
    "rewarded_ad_views": [rewarded_ad_views],
    "store_visits": [store_visits],
    "items_viewed_in_store": [items_viewed_in_store],
    "wishlist_items": [wishlist_items],
    "days_active_last_30": [days_active_last_30],
    "streak_days": [streak_days],
    "rage_quit_events": [rage_quit_events],
    "level_fail_rate": [level_fail_rate],
    "social_shares": [social_shares],
    "gender": [gender],
    "country": [country],
    "acquisition_channel": [acquisition_channel],
    "device_type": [device_type]
})

# Prediction button
if st.button("Predict Conversion"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
       
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ This player is likely to convert to a payer! (Probability: {proba[1]:.2%})")
        else:
            st.info(f"❌ This player is likely NOT to convert. (Probability: {proba[1]:.2%})")
       
        # Display probabilities
        st.write("**Class Probabilities:**")
        st.write(f"- Non‑payer: {proba[0]:.2%}")
        st.write(f"- Payer:      {proba[1]:.2%}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")