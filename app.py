import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import os
import base64

def predict_future(model, last_sequence, scaler=None, steps=30):
    """Predict future stock prices."""
    predictions = []
    for _ in range(steps):
        pred = model.predict(last_sequence.reshape(1, -1, 1))
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)

    if scaler:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    else:
        predictions = np.array(predictions).reshape(-1, 1)

    return predictions

if 'selected_company' not in st.session_state:
    st.session_state['selected_company'] = None

company_models = {
    "Apple": {
        "model": os.path.join("Trained Models", "Keras", "Apple_model.keras"),
        "last_sequence": np.load(os.path.join("Trained Models", "Last Sequences", "Apple_last_sequence.npy")),
        "logo": os.path.join("Logo", "Apple Logo.png")
    },
    "Coca Cola": {
        "model": os.path.join("Trained Models", "Keras", "COCO COLA_model.keras"),
        "last_sequence": np.load(os.path.join("Trained Models", "Last Sequences", "COCO_COLA_last_sequence.npy")),
        "logo": os.path.join("Logo", "Coca Cola.png")
    },
    "MasterCard": {
        "model": os.path.join("Trained Models", "Keras", "MasterCard_model.keras"),
        "last_sequence": np.load(os.path.join("Trained Models", "Last Sequences", "MasterCard_last_sequence.npy")),
        "logo": os.path.join("Logo", "mastercard logo.png")
    }
}


st.sidebar.subheader("Company Selection")
for company, details in company_models.items():
    st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="data:image/png;base64,{base64.b64encode(open(details['logo'], 'rb').read()).decode()}" 
             style="border-radius: 50%; width: 50px; height: 50px;">
        <p style="font-size: 14px; font-weight: bold;">{company}</p>
    </div>
    """,
    unsafe_allow_html=True
)

    if st.sidebar.button(company, key=f"sidebar_{company}"):
        st.session_state['selected_company'] = company


if st.session_state['selected_company']:
    selected_company = st.session_state['selected_company']
    st.title(f"{selected_company} Stock Prediction Dashboard")

   
    model_data = company_models[selected_company]
    try:
        model = load_model(model_data['model'])
        last_sequence = model_data['last_sequence']
    except Exception as e:
        st.error(f"Error loading data for {selected_company}: {e}")
        st.stop()

    
    st.image(model_data['logo'], width=100)
    st.markdown(f"### {selected_company}")

 
    num_days = st.slider("Prediction Days", min_value=1, max_value=30, value=7)

   
    predictions = predict_future(model, last_sequence, steps=num_days)

  
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=num_days)
    predicted_data = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions.flatten()})

    
    st.subheader(f"{selected_company} Predicted Line Chart")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=predicted_data['Date'], y=predicted_data['Predicted Close'], mode='lines', name='Predicted Close'))
    st.plotly_chart(fig_line)

  
    candlestick_data = pd.DataFrame({
        "Date": predicted_data["Date"],
        "Open": predicted_data["Predicted Close"] * 0.99,  
        "High": predicted_data["Predicted Close"] * 1.01,  
        "Low": predicted_data["Predicted Close"] * 0.98,   
        "Close": predicted_data["Predicted Close"]
    })

    
    st.subheader(f"{selected_company} Predicted Candlestick Chart")
    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=candlestick_data['Date'],
            open=candlestick_data['Open'],
            high=candlestick_data['High'],
            low=candlestick_data['Low'],
            close=candlestick_data['Close'],
            name='Predicted Candlestick'
        )
    ])
    st.plotly_chart(fig_candle)

 
    st.subheader("Predicted Values")
    st.write(predicted_data)

else:
    st.warning("Please select a company to proceed.")
