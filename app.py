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
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker} in the specified date range. Please check the ticker symbol or date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_resource # Use st.cache_resource for models if they are large
def load_trained_model(model_path="model/lstm_stock_model.h5"):
    """Loads the pre-trained LSTM model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"Could not load pre-trained model from {model_path}. You might need to train it first or check the path. Error: {e}")
        return None

# --- Stock Prediction Function (Integrate your actual logic here) ---
def predict_stock_prices(data, forecast_horizon, scaler, model):
    """
    Predicts future stock prices using the trained LSTM model.
    This is a placeholder. You'll need to replace this with your actual
    data preparation, prediction, and inverse scaling logic.
    """
    if data is None or model is None:
        return None, None

    # Example: Using 'Close' prices for prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaled_data = scaler.transform(close_prices)

    # Prepare data for LSTM (create sequences)
    # This part needs to match how your model was trained
    # For simplicity, let's assume the last N days are used for prediction
    n_lookback = 60 # This should match your model's input_shape
    
    if len(scaled_data) < n_lookback:
        st.warning(f"Not enough data to create sequences for prediction. Need at least {n_lookback} days.")
        return None, None

    last_n_days = scaled_data[-n_lookback:]
    X_test = np.array([last_n_days])

    # Make predictions
    predicted_scaled_prices = []
    current_batch = X_test[0] # The last sequence from our historical data

    for i in range(forecast_horizon):
        # Reshape for LSTM input: (1, n_lookback, 1)
        current_batch_reshaped = current_batch.reshape((1, n_lookback, 1))
        
        # Predict next day's price
        next_prediction = model.predict(current_batch_reshaped)[0]
        predicted_scaled_prices.append(next_prediction)
        
        # Update current_batch: remove first element, add new prediction
        current_batch = np.append(current_batch[1:], next_prediction, axis=0)

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
                # Initialize and fit the scaler on the 'Close' prices
                scaler = MinMaxScaler(feature_range=(0,1))
                # Fit scaler on the entire historical 'Close' data
                scaler.fit(df['Close'].values.reshape(-1, 1)) 
                
                # 3. Load or (re-train if needed) the model
                # For deployment, load a pre-trained model.
                # In a real scenario, you'd save your model after training.
                model = load_trained_model() # Assuming your model is in 'model/lstm_stock_model.h5'

                if model:
                    # 4. Make predictions
                    predicted_df, historical_close_prices = predict_stock_prices(df, forecast_horizon, scaler, model)

                    if predicted_df is not None:
                        st.subheader(f"Historical vs. Predicted Prices for {ticker_symbol}")

                        # Combine historical and predicted for visualization
                        historical_df = pd.DataFrame(historical_close_prices, index=df.index, columns=['Actual Close'])
                        
                        # Adjust index for plotting to avoid gaps
                        full_df = pd.concat([historical_df, predicted_df])

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
