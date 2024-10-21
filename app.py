import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Earthquake Prediction App", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #FFFFFF, #F0F8FF);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #4E7FDA, #39CCCC);
    }
    h1 {
        color: #4E7FDA;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4E7FDA;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #39CCCC;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("database.csv")
    data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
    print("Min date:", data['Date'].min())
    print("Max date:", data['Date'].max())
    return data

def safe_date_to_timestamp(date_str, time_str):
    try:
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", '%m/%d/%Y %H:%M:%S')
        return (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    except ValueError:
        return np.nan

def preprocess_data(data):
    data['Timestamp'] = data.apply(lambda row: safe_date_to_timestamp(row['Date'], row['Time']), axis=1)
    final_data = data.drop(['Date', 'Time'], axis=1)
    final_data = final_data.dropna()
    return final_data

@st.cache_resource
def create_and_train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def prediction_tab():
    st.header("🔮 Earthquake Prediction")
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
        longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)

    with col2:
        date = st.date_input("Date")
        time = st.time_input("Time")

    timestamp = datetime.datetime.combine(date, time)
    unix_time = (timestamp - datetime.datetime(1970, 1, 1)).total_seconds()

    if st.button("Predict Earthquake"):
        input_data = np.array([[unix_time, latitude, longitude]])
        scaled_input = st.session_state['scaler_X'].transform(input_data)
        scaled_prediction = st.session_state['model'].predict(scaled_input)
        prediction = st.session_state['scaler_y'].inverse_transform(scaled_prediction)
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Magnitude", f"{prediction[0][0]:.2f}")
        
        with col2:
            st.metric("Predicted Depth", f"{prediction[0][1]:.2f} km")
        
        fig = go.Figure(go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode='markers',
            marker=go.scattermapbox.Marker(size=14),
            text=[f"Magnitude: {prediction[0][0]:.2f}<br>Depth: {prediction[0][1]:.2f} km"]
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=latitude, lon=longitude), zoom=3),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def historical_data_tab(data):
    st.header("📜 Historical Earthquake Data")

    # Convert timestamp to datetime for min/max date selection
    data['datetime'] = pd.to_datetime(data['Timestamp'], unit='s')
    min_date = data['datetime'].min().date()
    max_date = data['datetime'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Convert dates to timestamps
    start_timestamp = pd.Timestamp(start_date).timestamp()
    end_timestamp = pd.Timestamp(end_date).timestamp() + 86400  # Add 24 hours to include end date

    filtered_data = data[(data['Timestamp'] >= start_timestamp) & 
                        (data['Timestamp'] <= end_timestamp)]

    st.subheader("Earthquake Locations")
    fig = px.scatter_mapbox(filtered_data, lat="Latitude", lon="Longitude", 
                            color="Magnitude", size="Magnitude",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            zoom=1, height=600)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Earthquake Data")
    st.dataframe(filtered_data)

def trends_tab(data):
    st.header("📈 Earthquake Trends")

    st.subheader("Earthquake Frequency Over Time")
    data['Date'] = pd.to_datetime(data['Timestamp'], unit='s')
    monthly_freq = data.resample('M', on='Date').size().reset_index(name='Frequency')
    fig = px.line(monthly_freq, x='Date', y='Frequency')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Magnitude Distribution")
    fig = px.histogram(data, x="Magnitude", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Depth vs Magnitude")
    fig = px.scatter(data, x="Depth", y="Magnitude", color="Magnitude", 
                     color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Earthquake Hotspots")
    fig = px.density_mapbox(data, lat='Latitude', lon='Longitude', z='Magnitude', 
                            radius=10, center=dict(lat=0, lon=180), zoom=0,
                            mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

def model_performance_tab(history):
    st.header("📈 Model Training Performance")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'],
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'],
        name='Validation Loss'
    ))
    fig.update_layout(title='Training History',
                     xaxis_title='Epoch',
                     yaxis_title='Loss')
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🌍 Earthquake Prediction and Analysis App")

    data = load_data()
    final_data = preprocess_data(data)
    
    X = final_data[['Timestamp', 'Latitude', 'Longitude']].values
    y = final_data[['Magnitude', 'Depth']].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    model, history = create_and_train_model(X_train, y_train, X_val, y_val)
    
    st.session_state['scaler_X'] = scaler_X
    st.session_state['scaler_y'] = scaler_y
    st.session_state['model'] = model

    st.sidebar.header("📊 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis", 
        ["Prediction", "Historical Data", "Trends", "Model Performance"]
    )

    if analysis_type == "Prediction":
        prediction_tab()
    elif analysis_type == "Historical Data":
        historical_data_tab(final_data)
    elif analysis_type == "Trends":
        trends_tab(final_data)
    else:
        model_performance_tab(history)

if __name__ == "__main__":
    main()