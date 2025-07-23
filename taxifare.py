import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split




with open("pipeline2.pkl",'rb') as file:
    loaded_pipeline = pickle.load(file)

st.title("Taxi fare Prediction")
st.write("Enter trip details to predict the total fare amount:")
passenger_count = st.number_input('Passenger Count(1 to 6):', min_value=1, max_value=10, value=1)
RatecodeID = st.selectbox('RatecodeID', [1, 2, 3, 4, 5, 6])
trip_distance_km = st.number_input('Trip Distance (miles)', min_value=0.0, value=2.0)
payment_type = st.selectbox('Payment Type', [1, 2, 3, 4])
tip_amount = st.number_input('Tip Amount', min_value=0.0, value=0.0)
hour = st.slider('Hour (0-23)', 0, 23, 12)


if st.button('Predict Fare'):
  input_data = {
        'passenger_count': passenger_count,
        'trip_distance_km': trip_distance_km,
        'RatecodeID': RatecodeID,
        'payment_type': payment_type,
        'tip_amount': tip_amount,
        'hour': hour,
        
    }
  data_taxi=pd.DataFrame(input_data,index=[0])
  prediction = loaded_pipeline.predict(data_taxi)
  st.success(f"Predicted Total Fare: ${prediction[0]:.2f}")