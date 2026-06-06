import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# --- Title and Description ---
st.title("📈 AI Stock Price Prediction using LSTM")
st.markdown("""
    This application utilizes Long Short-Term Memory (LSTM) neural networks to forecast future stock prices
    based on historical trends. Enter a stock ticker, select a date range, and a forecast horizon to see predictions.
""")

# --- Caching Data Loading and Model Loading ---
@st.cache_data
def get_historical_data(ticker, start_date, end_date):
    """Fetches historical stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            st.error(f"No data found for {ticker} in the specified date range. Please check the ticker symbol or date range.")
            return None
        
        # Flatten MultiIndex columns if returned by newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_resource 
def load_trained_model(model_path="lstm_stock_model.h5"):
    """Loads the pre-trained LSTM model from the root directory path."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"Could not load pre-trained model from {model_path}. Error: {e}")
        return None

# --- Stock Prediction Function ---
def predict_stock_prices(data, forecast_horizon, scaler, model):
    """Predicts future stock prices using the trained LSTM model."""
    if data is None or model is None:
        return None, None

    # Target closing price dynamically
    close_col = 'Close' if 'Close' in data.columns else data.columns
    close_prices = data[close_col].values.reshape(-1, 1)

    # Scale the data
    scaled_data = scaler.transform(close_prices)

    # Automatically extract the exact lookback window from your model's input layer
    try:
        n_lookback = model.input_shape[1]
    except Exception:
        n_lookback = 60 # Fallback default
    
    if len(scaled_data) < n_lookback:
        st.warning(f"Not enough data to create sequences for prediction. Need at least {n_lookback} days.")
        return None, None

    # Extract the last available window slice
    last_n_days = scaled_data[-n_lookback:]
    current_batch = last_n_days.reshape((1, n_lookback, 1))
    
    predicted_scaled_prices = []

    for i in range(forecast_horizon):
        # Predict next day's price
        next_prediction = model.predict(current_batch, verbose=0)
        
        # Extract scalar value from prediction array safely
        pred_scalar = next_prediction[0, 0]
        predicted_scaled_prices.append([pred_scalar])
        
        # Create 3D matrix piece for appending: (1, 1, 1)
        next_pred_reshaped = np.array([[[pred_scalar]]])
        
        # Slide lookback array window forward on Axis 1
        current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)

    # Inverse transform values back to stock dollars
    predicted_prices = scaler.inverse_transform(np.array(predicted_scaled_prices))

    # Generate dates tracking sequentially from the last historical point
    last_date = pd.to_datetime(data.index[-1])
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, forecast_horizon + 1)]

    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Close'])
    return predicted_df, close_prices


# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=365*3) # 3 years back
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=today)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)

if st.sidebar.button("Predict"):
    if start_date >= end_date:
        st.sidebar.error("Error: Start date must be before end date.")
    else:
        with st.spinner(f"Fetching data and predicting for {ticker_symbol}..."):
            df = get_historical_data(ticker_symbol, start_date, end_date)

            if df is not None:
                close_col = 'Close' if 'Close' in df.columns else df.columns
                scaler = MinMaxScaler(feature_range=(0,1))
                scaler.fit(df[close_col].values.reshape(-1, 1)) 
                
                model = load_trained_model("lstm_stock_model.h5") 

                if model:
                    # Debug notification to check model configuration on screen
                    st.sidebar.info(f"Model Input Shape Detected: {model.input_shape}")
                    
                    predicted_df, historical_close_prices = predict_stock_prices(df, forecast_horizon, scaler, model)

                    if predicted_df is not None:
                        st.subheader(f"Historical vs. Predicted Prices for {ticker_symbol}")

                        historical_df = pd.DataFrame(historical_close_prices, index=df.index, columns=['Actual Close'])
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(historical_df.index, historical_df['Actual Close'], label='Actual Prices', color='blue')
                        ax.plot(predicted_df.index, predicted_df['Predicted Close'], label='Predicted Prices', color='red', linestyle='--')
                        
                        ax.set_title(f'{ticker_symbol} Stock Price Prediction')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price (USD)')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        st.subheader("Predicted Prices Table")
                        st.dataframe(predicted_df)
                    else:
                        st.warning("Prediction could not be generated. Check data processing steps.")
                else:
                    st.error("LSTM model could not be loaded from root directory.")
            else:
                st.error("Failed to retrieve historical data.")

st.markdown("---")
st.markdown("Project by: [T.A.SRINIVAS](https://github.com)")
st.markdown("Tech Stack: Python, TensorFlow, Keras, yfinance, scikit-learn, pandas, numpy, matplotlib, Streamlit")
