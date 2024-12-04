# Mini_project_traffic
# 🚗 Accident Severity Prediction Platform

Welcome to the **Accident Severity Prediction Platform**, an innovative tool designed to predict accident severity using machine learning. The platform leverages weather data to classify accidents as **Light**, **Moderate**, or **Severe**, helping city planners, emergency responders, and insurance companies improve safety measures.

---

## 🚀 Features
- **Real-time Predictions:** Predict accident severity using live weather inputs.
- **Random Input Generation:** Automatically generate random weather data for testing.
- **Custom Inputs:** Input your own weather data for personalized predictions.
- **Interactive Dashboard:** Learn about the model and the features it uses.

---

## 🛠️ How It Works
1. Input weather data such as temperature, humidity, wind speed, and precipitation.
2. The model, built with a **Random Forest Classifier**, predicts accident severity based on the input conditions.
3. Outputs one of the following severities:
   - **Light**: Low severity accident.
   - **Moderate**: Medium severity accident.
   - **Severe**: High severity accident.

---

## 📊 Model Overview
- **Algorithm:** Random Forest Classifier
- **Features Used:**
  - Temperature (°F)
  - Wind Chill (°F)
  - Humidity (%)
  - Pressure (in)
  - Visibility (mi)
  - Wind Speed (mph)
  - Precipitation (in)
  - Wind Direction (North/South)
  - Weather Conditions (Clear/Rain)

---

## 📂 Repository Contents
- **`app.py`**: Streamlit-based web application for accident severity prediction.
- **`accident_train.csv`**: Dataset used for training the model.
- **`Rforest.ipynb`**: Jupyter Notebook containing model training and evaluation code.
- **`random_forest_model.pkl`**: Pre-trained Random Forest model.
- **`accident_image.jpg`**: Image displayed in the application interface.

---

## 🖥️ Requirements
- Python 3.8 or above
- Libraries: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `Pillow`

---
