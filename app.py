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
        
        # Flatten MultiIndex columns returned by newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_resource 
def load_trained_model(model_path="model/lstm_stock_model.h5"):
    """Loads the pre-trained LSTM model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"Could not load pre-trained model from {model_path}. You might need to train it first or check the path. Error: {e}")
        return None

# --- Stock Prediction Function ---
def predict_stock_prices(data, forecast_horizon, scaler, model):
    """Predicts future stock prices using the trained LSTM model."""
    if data is None or model is None:
        return None, None

    # Handle close price array generation safely
    close_col = 'Close' if 'Close' in data.columns else data.columns
    close_prices = data[close_col].values.reshape(-1, 1)

    # Scale the data
    scaled_data = scaler.transform(close_prices)

    n_lookback = 60 
    
    if len(scaled_data) < n_lookback:
        st.warning(f"Not enough data to create sequences for prediction. Need at least {n_lookback} days.")
        return None, None

    last_n_days = scaled_data[-n_lookback:]
    
    # Reshape explicitly to 3D batch: (samples, time_steps, features) -> (1, 60, 1)
    current_batch = last_n_days.reshape((1, n_lookback, 1))
    predicted_scaled_prices = []

    for i in range(forecast_horizon):
        # Predict next step value -> shape output: (1, 1)
        next_prediction = model.predict(current_batch, verbose=0)
        predicted_scaled_prices.append(next_prediction)
        
        # Reshape prediction to match 3D tensor layout (1, 1, 1)
        next_pred_reshaped = next_prediction.reshape((1, 1, 1))
        
        # Slide window array forward on axis 1
        current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)

    # Inverse transform to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_scaled_prices).reshape(-1, 1))

    # Generate future dates for the predictions
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, forecast_horizon + 1)]

    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Close'])
    return predicted_df, close_prices


# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=365*3) # 3 years ago
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=today)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)

if st.sidebar.button("Predict"):
    if start_date >= end_date:
        st.sidebar.error("Error: Start date must be before end date.")
    else:
        with st.spinner(f"Fetching data and predicting for {ticker_symbol}..."):
            # 1. Collect historical stock data
            df = get_historical_data(ticker_symbol, start_date, end_date)

            if df is not None:
                # 2. Preprocess and scale data
                close_col = 'Close' if 'Close' in df.columns else df.columns
                scaler = MinMaxScaler(feature_range=(0,1))
                scaler.fit(df[close_col].values.reshape(-1, 1)) 
                
                # 3. Load model asset
                model = load_trained_model() 

                if model:
                    # 4. Make predictions
                    predicted_df, historical_close_prices = predict_stock_prices(df, forecast_horizon, scaler, model)

                    if predicted_df is not None:
                        st.subheader(f"Historical vs. Predicted Prices for {ticker_symbol}")

                        # Combine historical and predicted for visualization
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
                        st.warning("Prediction could not be generated. Check model and data processing steps.")
                else:
                    st.error("LSTM model could not be loaded. Please ensure it's saved correctly and the path is accurate.")
            else:
                st.error("Failed to retrieve historical data. Please check ticker or date range.")

st.markdown("---")
st.markdown("Project by: [T.A.SRINIVAS/https://github.com/srinivasta](https://github.com/srinivasta)")
st.markdown("Tech Stack: Python, TensorFlow, Keras, yfinance, scikit-learn, pandas, numpy, matplotlib, Streamlit")
