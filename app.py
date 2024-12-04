import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image

# Load pre-trained model (ensure the model file is in the same directory)
model = joblib.load("random_forest_model.pkl")

# Function to generate random input
def generate_random_input():
    random_input = {
        'Temperature(F)': random.uniform(-30, 110),
        'Wind_Chill(F)': random.uniform(-50, 100),
        'Humidity(%)': random.randint(0, 100),
        'Pressure(in)': random.uniform(28, 31),
        'Visibility(mi)': random.uniform(1, 20),
        'Wind_Speed(mph)': random.uniform(0, 60),
        'Precipitation(in)': random.uniform(0, 1),
        'Wind_Direction_N': random.choice([0, 1]),
        'Wind_Direction_S': random.choice([0, 1]),
        'Weather_Condition_Clear': random.choice([0, 1]),
        'Weather_Condition_Rain': random.choice([0, 1])
    }
    return random_input

# Function to preprocess input data (convert it to DataFrame format)
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
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
    st.write("""
        Welcome to the **Accident Severity Prediction Platform**. This platform uses machine learning to predict the severity of accidents
        based on weather conditions such as temperature, humidity, wind speed, and more. It predicts whether the accident severity is **Light**, 
        **Moderate**, or **Severe** based on these conditions.
    """)
    
    # Display an image on the Home Page (make sure the image is in the same folder as app.py or specify path)
    img = Image.open("accident_image.jpg")  # Replace with your image file
    st.image(img, caption="Accident Severity Prediction Model", use_container_width=True)

    
    st.write("""
        The model is trained using weather data and can predict the severity of an accident based on different factors. 
        In the following sections, you can learn more about how the model works and try predicting accident severity using random inputs or custom data.
    """)
    
    st.markdown("""
        **How It Works:**
        1. Input weather data such as temperature, wind speed, and humidity.
        2. The model will predict the severity of the accident based on the input conditions.
        3. The severity can be classified as **Light**, **Moderate**, or **Severe**.
    """)

# --- DASHBOARD PAGE ---
elif page == "Dashboard":
    st.title("📊 Dashboard - About the Model")
    
    st.write("""
        **Model Overview:**
        The **Accident Severity Prediction Model** uses various weather features to determine the severity of an accident. 
        Features such as temperature, humidity, wind speed, and precipitation are fed into a **Random Forest Classifier** to predict one of the following outcomes:
    """)
    
    st.write("""
        - **Light**: Low severity accident.
        - **Moderate**: Medium severity accident.
        - **Severe**: High severity accident.
        - **Unknown**: Prediction could not be determined due to lack of data or conflicting input.
    """)
    
    st.markdown("""
        **Model's Features:**
        - **Temperature (°F)**
        - **Wind Chill (°F)**
        - **Humidity (%)**
        - **Pressure (in)**
        - **Visibility (mi)**
        - **Wind Speed (mph)**
        - **Precipitation (in)**
        - **Wind Direction (N/S)**
        - **Weather Condition (Clear/Rain)**
    """)
    
    st.write("This model can help predict the severity of accidents, aiding in better decision-making during emergency responses.")

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.title("🔮 Predict Accident Severity")
    
    # Generate random data button
    if st.button("Generate Random Data & Predict Severity"):
        random_input = generate_random_input()

        # Display random input data
        st.write("### Random Input Data")
        for feature, value in random_input.items():
            st.write(f"{feature}: {value:.2f}")

        # Predict severity based on random input
        severity = predict_severity(random_input)

        # Display Prediction Result
        st.success(f"**Predicted Accident Severity: {severity}**")
    
    # Option for user to input custom data
    st.write("### Or, input your own data:")
    
    temp = st.number_input("Temperature (°F)", value=70.0)
    wind_chill = st.number_input("Wind Chill (°F)", value=65.0)
    humidity = st.number_input("Humidity (%)", value=50)
    pressure = st.number_input("Pressure (in)", value=30.0)
    visibility = st.number_input("Visibility (mi)", value=10.0)
    wind_speed = st.number_input("Wind Speed (mph)", value=15)
    precipitation = st.number_input("Precipitation (in)", value=0.1)
    wind_direction_n = st.selectbox("Wind Direction North", [0, 1])
    wind_direction_s = st.selectbox("Wind Direction South", [0, 1])
    weather_condition_clear = st.selectbox("Weather Condition Clear", [0, 1])
    weather_condition_rain = st.selectbox("Weather Condition Rain", [0, 1])

    # Collect user inputs
    custom_input = {
        'Temperature(F)': temp,
        'Wind_Chill(F)': wind_chill,
        'Humidity(%)': humidity,
        'Pressure(in)': pressure,
        'Visibility(mi)': visibility,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precipitation,
        'Wind_Direction_N': wind_direction_n,
        'Wind_Direction_S': wind_direction_s,
        'Weather_Condition_Clear': weather_condition_clear,
        'Weather_Condition_Rain': weather_condition_rain
    }
    
    if st.button("Predict Custom Data Severity"):
        severity = predict_severity(custom_input)
        st.success(f"**Predicted Accident Severity: {severity}**")
