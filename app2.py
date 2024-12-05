import streamlit as st
import random
import pandas as pd
import joblib
from PIL import Image

# Load pre-trained model (ensure the model file is in the same directory)
model = joblib.load("traffic_model.pkl")

# Function to generate random input with severity option
def generate_random_input():
    random_input = {
        'Temperature(F)': random.uniform(-30, 110),
        'Wind_Chill(F)': random.uniform(-50, 100),
        'Distance(mi)':random.uniform(0,100),
        'Humidity(%)': random.randint(0, 100),
        'Pressure(in)': random.uniform(28, 31),
        'Visibility(mi)': random.uniform(1, 20),
        'Wind_Speed(mph)': random.uniform(0, 60),
        'Precipitation(in)': random.uniform(0, 1),
        'Amenity': random.choice([0, 1]),
        'Bump': random.choice([0, 1]),
        'Crossing': random.choice([0, 1]),
        'Junction': random.choice([0, 1]),
        'Station': random.choice([0, 1]),
        'Stop': random.choice([0, 1]),
        'Traffic_Signal': random.choice([0, 1]),
        'Sunrise_Sunset': random.choice([0, 1]),
        'Wind_Direction_CALM': random.choice([0, 1]),
        'Wind_Direction_Calm': random.choice([0, 1]),
        'Wind_Direction_E': random.choice([0, 1]),
        'Wind_Direction_ENE': random.choice([0, 1]),
        'Wind_Direction_ESE': random.choice([0, 1]),
        'Wind_Direction_East': random.choice([0, 1]),
        'Wind_Direction_N': random.choice([0, 1]),
        'Wind_Direction_NE': random.choice([0, 1]),
        'Wind_Direction_NNE': random.choice([0, 1]),
        'Wind_Direction_NNW': random.choice([0, 1]),
        'Wind_Direction_NW': random.choice([0, 1]),
        'Wind_Direction_North': random.choice([0, 1]),
        'Wind_Direction_S': random.choice([0, 1]),
        'Wind_Direction_SE': random.choice([0, 1]),
        'Wind_Direction_SSE': random.choice([0, 1]),
        'Wind_Direction_SSW': random.choice([0, 1]),
        'Wind_Direction_SW': random.choice([0, 1]),
        'Wind_Direction_South': random.choice([0, 1]),
        'Wind_Direction_VAR': random.choice([0, 1]),
        'Wind_Direction_Variable': random.choice([0, 1]),
        'Wind_Direction_W': random.choice([0, 1]),
        'Wind_Direction_WNW': random.choice([0, 1]),
        'Wind_Direction_WSW': random.choice([0, 1]),
        'Wind_Direction_West': random.choice([0, 1]),
        'Weather_Condition_Clear': random.choice([0, 1]),
        'Weather_Condition_Cloudy': random.choice([0, 1]),
        'Weather_Condition_Fog': random.choice([0, 1]),
        'Weather_Condition_Freezing Conditions': random.choice([0, 1]),
        'Weather_Condition_Rain': random.choice([0, 1]),
        'Weather_Condition_Snow': random.choice([0, 1]),
        'Weather_Condition_Thunderstorm': random.choice([0, 1]),
        'Weather_Condition_Unknown': random.choice([0, 1])
    }
    return random_input

# Function to preprocess input data (convert it to DataFrame format)
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Get the feature names from the trained model to align input columns
    trained_features = model.feature_names_in_
    
    # Reindex the input DataFrame to match the model's expected feature names
    input_df = input_df.reindex(columns=trained_features, fill_value=0)
    
    return input_df

# Function to predict severity
def predict_severity(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    severity_map = {1: 'Light', 2: 'Moderate', 3: 'Severe', 4: 'Unknown'}
    severity = severity_map.get(prediction[0], 'Unknown')
    return severity

# Streamlit UI

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚗", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Select a Page", ["Home", "Dashboard", "Prediction"])

# --- HOME PAGE ---
if page == "Home":
    st.title("🚗 Accident Severity Prediction Platform")
    st.write("""Welcome to the *Accident Severity Prediction Platform*. This platform uses machine learning to predict the severity of accidents
                based on weather conditions such as temperature, humidity, wind speed, and more. It predicts whether the accident severity is *Light*, 
                *Moderate, or **Severe* based on these conditions.""")
    
    # Display an image on the Home Page (make sure the image is in the same folder as app.py or specify path)
    img = Image.open("accident_image.jpg")  # Replace with your image file
    st.image(img, caption="Accident Severity Prediction Model", use_container_width=True)

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.title("🔮 Predict Accident Severity")
    
    # # Select severity for random input
    # severity_option = st.selectbox("Select Severity Level for Random Input", ["Light", "Moderate", "Severe"])
    
    # Button to generate random data
    if st.button("Generate Random Input"):
        random_input = generate_random_input()
        
        # Display random input data
        st.write("### Random Input Data")
        for feature, value in random_input.items():
            if isinstance(value, (int, float)):  # Check if the value is numeric
                st.write(f"{feature}: {value:.6f}")
            else:
                st.write(f"{feature}: {value}")
        
        # Predict severity based on random input
        severity = predict_severity(random_input)

        # Display Prediction Result
        st.success(f"*Predicted Accident Severity: {severity}*")
    
    # Option for user to input custom data
    st.write("### Or, input your own data:")
    temp = st.number_input("Temperature (°F)", value=70.0)
    wind_chill = st.number_input("Wind Chill (°F)", value=65.0)
    humidity = st.number_input("Humidity (%)", value=50)
    pressure = st.number_input("Pressure (in)", value=30.0)
    visibility = st.number_input("Visibility (mi)", value=10.0)
    wind_speed = st.number_input("Wind Speed (mph)", value=15)
    precipitation = st.number_input("Precipitation (in)", value=0.1)
    wind_direction = st.selectbox("Wind Direction", ["North", "South", "East", "West"])
    weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
    amenity = st.checkbox("Amenity Present")
    bump = st.checkbox("Bump Present")
    crossing = st.checkbox("Crossing Present")
    junction = st.checkbox("Junction Present")
    station = st.checkbox("Station Nearby")
    stop = st.checkbox("Stop Nearby")
    traffic_signal = st.checkbox("Traffic Signal Present")
    sunrise_sunset = st.checkbox("Sunrise or Sunset")
    
    # New latitude and longitude inputs
    start_lat = st.number_input("Start Latitude", format="%.6f", value=0.0)
    start_lng = st.number_input("Start Longitude", format="%.6f", value=0.0)
    
    # Collect user inputs
    custom_input = {
        'Temperature(F)': temp,
        'Wind_Chill(F)': wind_chill,
        'Humidity(%)': humidity,
        'Pressure(in)': pressure,
        'Visibility(mi)': visibility,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precipitation,
        'Wind_Direction': wind_direction,
        'Weather_Condition': weather_condition,
        'Amenity': 1 if amenity else 0,
        'Bump': 1 if bump else 0,
        'Crossing': 1 if crossing else 0,
        'Junction': 1 if junction else 0,
        'Station': 1 if station else 0,
        'Stop': 1 if stop else 0,
        'Traffic_Signal': 1 if traffic_signal else 0,
        'Sunrise_Sunset': 1 if sunrise_sunset else 0,
        'Start_Latitude': start_lat,
        'Start_Longitude': start_lng
    }
    
    if st.button("Predict Custom Data Severity"):
        severity = predict_severity(custom_input)
        st.success(f"*Predicted Accident Severity: {severity}*")