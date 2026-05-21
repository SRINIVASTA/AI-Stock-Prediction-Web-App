# 📈 AI Stock Price Predictor using LSTM

[![Python Version](https://shields.io)](https://python.org)
[![Streamlit App](https://streamlit.io)](https://streamlit.io)
[![Framework](https://shields.io)](https://tensorflow.org)

An interactive, deep-learning-powered web application that forecasts future stock prices. The app utilizes **Long Short-Term Memory (LSTM)** neural networks to recognize patterns in historical financial data and predict future market trends.

---

## 🚀 Features

* **Real-time Data Fetching**: Retrieves live historical market data using the `yfinance` API.
* **Autoregressive Forecasting**: Generates multi-day future price predictions sequentially.
* **Interactive Controls**: Customise ticker symbols, historical data ranges, and the future forecast horizon (1–30 days) right from the sidebar.
* **Clean Financial Visualizations**: Interactive charting powered by `matplotlib` that filters out weekends for accurate calendar plotting.
* **Formatted Data Reports**: Displays raw predicted data tables cleanly formatted in USD currency.

---

## 🛠️ Tech Stack

* **Frontend/Deployment**: Streamlit
* **Deep Learning**: TensorFlow / Keras
* **Data Processing**: Pandas, NumPy, Scikit-Learn (MinMaxScaler)
* **Financial Data**: yfinance (Yahoo Finance API)
* **Data Visualization**: Matplotlib

---

## 📂 Project Structure

```text
├── model/
│   └── lstm_stock_model.h5   # Pre-trained Keras LSTM model weights
├── app.py                     # Main Streamlit application file
├── requirements.txt           # Environment deployment dependencies
└── README.md                  # Project documentation
```

---

## 💻 Local Installation & Setup

Follow these steps to run the web application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd ai-stock-prediction-web-app
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## ⚠️ Important Deployment Notes

* **Pre-trained Model**: Ensure your trained model file `lstm_stock_model.h5` is placed inside the `model/` directory before running or deploying.
* **Deployment Platforms**: If deploying to **Streamlit Cloud**, the build environment will automatically use the `requirements.txt` file to compile dependencies (including TensorFlow).
* **API Dependencies**: The app relies heavily on the `yfinance` package scraping mechanisms. If data fails to pull, run `pip install --upgrade yfinance` to pull down Yahoo's latest decryption patches.

---

## 👨‍💻 Project Developer

Developed with ❤️ by **[T.A.SRINIVAS](https://github.com)**. Feel free to reach out, star the repository, or open an issue if you encounter any bugs!
