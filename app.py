import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

model = load_model('model.h5')
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title('House Price Prediction Model')


#Streamlit app
with st.form("input_form"):
    id_val = st.number_input("ID", value=1234567890, step=1)
    date_val = st.date_input("Date")
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
    living_area = st.number_input("Living Area (sq ft)", min_value=0, step=1)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=0, step=1)
    floors = st.number_input("Number of Floors", min_value=0, step=1)
    waterfront = st.selectbox("Waterfront Present", [0, 1])
    views = st.number_input("Number of Views", min_value=0, step=1)
    condition = st.number_input("Condition of the House (1–5)", min_value=1, max_value=5, step=1)
    grade = st.number_input("Grade of the House (1–13)", min_value=1, max_value=13, step=1)
    area_excl_basement = st.number_input("Area of the House (Excl. Basement)", min_value=0, step=1)
    basement_area = st.number_input("Area of the Basement", min_value=0, step=1)
    built_year = st.number_input("Built Year", min_value=1800, max_value=2100, step=1)
    renov_year = st.number_input("Renovation Year", min_value=0, max_value=2100, step=1)
    postal_code = st.number_input("Postal Code", min_value=0, step=1)
    latitude = st.number_input("Latitude", format="%.6f")
    longitude = st.number_input("Longitude", format="%.6f")
    living_area_renov = st.number_input("Living Area after Renovation", min_value=0, step=1)
    lot_area_renov = st.number_input("Lot Area after Renovation", min_value=0, step=1)
    schools_nearby = st.number_input("Number of Schools Nearby", min_value=0, step=1)
    distance_airport = st.number_input("Distance from the Airport (km)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict Price")


input_df = pd.DataFrame({
        "number of bedrooms": [bedrooms],
        "number of bathrooms": [bathrooms],
        "living area": [living_area],
        "lot area": [lot_area],
        "number of floors": [floors],
        "waterfront present": [waterfront],
        "number of views": [views],
        "condition of the house": [condition],
        "grade of the house": [grade],
        "Area of the house(excluding basement)": [area_excl_basement],
        "Area of the basement": [basement_area],
        "Built Year": [built_year],
        "Renovation Year": [renov_year],
        "Lattitude": [latitude],
        "Longitude": [longitude],
        "living_area_renov": [living_area_renov],
        "lot_area_renov": [lot_area_renov],
        "Number of schools nearby": [schools_nearby],
        "Distance from the airport": [distance_airport]
    })
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

st.success(f"Predicted Price: Rs. {prediction[0]:.2f}")