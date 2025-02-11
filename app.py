# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# ARIMA model
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Prophet model
from prophet import Prophet

# For LSTM forecasting
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler

# For interactive Plotly charts
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# --- Custom Theming and Layout Improvements ---
st.markdown(
    """
    <style>
    /* Set a light background for the app */
    .reportview-container {
        background: #f0f2f6;
    }
    /* Style the main content container */
    .main {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    /* Custom header styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Updated Home Dashboard ---
st.title("Time Series Forecasting")
st.markdown(
    """
    <div style="background-color: #e8f0fe; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;">Welcome!</h2>
        <p style="font-size: 1.1rem;">
            Explore our advanced forecasting models to predict stock closing prices:
        </p>
        <ul style="font-size: 1.1rem;">
            <li><strong>ARIMA:</strong> Ideal for linear time series with clear interpretability.</li>
            <li><strong>Prophet:</strong> Developed by Facebook, it handles trends and seasonality even with missing data.</li>
            <li><strong>LSTM:</strong> A deep learning model capable of capturing complex non-linear patterns.</li>
        </ul>
        <p style="font-size: 1.1rem;">
            <strong>Data Source:</strong> All data is sourced live from Yahoo Finance using <code>yfinance</code>.
            You can also try a Kaggle dataset, for example, 
            <a href="https://www.kaggle.com/datasets/rohitsahoo/historical-stock-prices" target="_blank">Historical Stock Prices</a>.
        </p>
        <p style="font-size: 1.1rem;">
            Use the sidebar to configure data selections, choose your forecasting model, and adjust visualization settings. 
            Enjoy exploring and forecasting your time series data!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Sidebar Configuration with Expanders ---
st.sidebar.header("Configuration")

with st.sidebar.expander("1. Data Selection", expanded=True):
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
    end_date = st.date_input("End Date", value=datetime.today())
    forecast_period = st.number_input("Forecast Period (Days)", min_value=1, value=30)

with st.sidebar.expander("2. Model Selection", expanded=True):
    model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "Prophet", "LSTM"])
    
    if model_choice == "ARIMA":
        st.markdown("**ARIMA Parameters**")
        p = st.number_input("AR(p)", min_value=0, value=5)
        d = st.number_input("I(d)", min_value=0, value=1)
        q = st.number_input("MA(q)", min_value=0, value=0)
        order = (p, d, q)
    elif model_choice == "LSTM":
        st.markdown("**LSTM Parameters**")
        st.info("Using pre-trained model. Adjust 'Look-back Period' if necessary.")
        look_back = st.number_input("Look-back Period", min_value=10, value=60)

with st.sidebar.expander("3. Visualization Options", expanded=True):
    use_interactive = st.checkbox("Use interactive charts (Plotly)", value=False)
    show_decomposition = st.checkbox("Show seasonal decomposition", value=False)
    st.markdown("### Visualization Date Range")
    viz_start_date = st.date_input("Visualization Start Date", value=start_date)
    viz_end_date = st.date_input("Visualization End Date", value=end_date)

# --- Function to Load Data ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # If the DataFrame has MultiIndex columns, flatten them:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    # Ensure the 'Close' column is 1-dimensional:
    if isinstance(data['Close'], pd.DataFrame):
        data['Close'] = data['Close'].iloc[:, 0]
    return data

# --- ARIMA Forecasting Function ---
def run_arima(data, order, forecast_steps):
    ts = data.set_index("Date")["Close"]
    model = sm.tsa.ARIMA(ts, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1), periods=forecast_steps, freq='B')
    forecast_series = pd.Series(forecast, index=forecast_index)
    return forecast_series

# --- Prophet Forecasting Function ---
def run_prophet(data, forecast_period):
    prophet_df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)
    return forecast

# --- LSTM Forecasting Function (Using Pre-trained Model) ---
def run_lstm(data, forecast_steps, look_back):
    try:
        lstm_model = load_model("lstm_model.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error("Error loading pre-trained LSTM model and scaler. Ensure 'lstm_model.h5' and 'scaler.pkl' are in the app directory.")
        return None

    dataset = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(dataset)
    
    last_sequence = scaled_data[-look_back:]
    current_seq = last_sequence.reshape(1, look_back, 1)
    
    predictions = []
    for _ in range(forecast_steps):
        pred = lstm_model.predict(current_seq)[0,0]
        predictions.append(pred)
        current_seq = np.append(current_seq[0,1:], [[pred]], axis=0)
        current_seq = current_seq.reshape(1, look_back, 1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = pd.to_datetime(data['Date'].iloc[-1])
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='B')
    forecast_series = pd.Series(predictions.flatten(), index=forecast_index)
    return forecast_series

# --- Function to Create Seasonal Decomposition Plot ---
def seasonal_decomposition_plot(data):
    ts = data.set_index("Date")["Close"]
    # Set frequency to business days:
    ts = ts.asfreq('B')
    # Fill missing values (if any) using forward fill:
    ts = ts.fillna(method='ffill')
    # Now perform seasonal decomposition:
    result = seasonal_decompose(ts, model='additive', period=252)  # Approximate yearly seasonality for trading days
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    result.observed.plot(ax=axs[0], legend=False)
    axs[0].set_ylabel("Observed")
    result.trend.plot(ax=axs[1], legend=False)
    axs[1].set_ylabel("Trend")
    result.seasonal.plot(ax=axs[2], legend=False)
    axs[2].set_ylabel("Seasonal")
    result.resid.plot(ax=axs[3], legend=False)
    axs[3].set_ylabel("Residual")
    plt.xlabel("Date")
    plt.tight_layout()
    return fig

# --- Download Forecast as CSV ---
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# --- Main App Logic ---
if st.sidebar.button("Load Data"):
    with st.spinner("Loading data..."):
        data = load_data(ticker, start_date, end_date)
    st.success("Data loaded successfully!")
    
    # Filter data for visualization based on user-selected date range
    filtered_data = data[(data['Date'].dt.date >= viz_start_date) & (data['Date'].dt.date <= viz_end_date)]
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Historical Data", "Forecast", "Additional Analysis"])
    
    with tab1:
        st.subheader(f"Historical Data for {ticker}")
        st.dataframe(filtered_data)
        
        # Historical chart using the filtered data
        if use_interactive:
            fig = px.line(filtered_data, x='Date', y='Close', title=f"{ticker} Historical Closing Prices")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(filtered_data['Date'], filtered_data['Close'], label="Close Price", color='blue')
            ax.set_title(f"{ticker} Historical Closing Prices")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
    
    with tab2:
        st.header(f"Forecast using {model_choice}")
        if model_choice == "ARIMA":
            forecast_series = run_arima(data, order, forecast_period)
            st.subheader("ARIMA Forecast")
            st.write(forecast_series)
        elif model_choice == "Prophet":
            forecast_df = run_prophet(data, forecast_period)
            st.subheader("Prophet Forecast (Last Few Predictions)")
            st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))
            # Extract forecast line
            forecast_series = pd.Series(forecast_df['yhat'].tail(forecast_period).values,
                                        index=pd.to_datetime(forecast_df['ds'].tail(forecast_period)))
        elif model_choice == "LSTM":
            forecast_series = run_lstm(data, forecast_period, look_back)
            if forecast_series is not None:
                st.subheader("LSTM Forecast")
                st.write(forecast_series)
        
        if forecast_series is not None:
            # Display forecast chart using the user-selected visualization window for historical data
            if use_interactive:
                df_forecast = pd.DataFrame({"Date": forecast_series.index, "Forecast": forecast_series.values})
                fig = px.line(df_forecast, x="Date", y="Forecast", title=f"{model_choice} Forecast")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                if not filtered_data.empty:
                    ax.plot(filtered_data['Date'], filtered_data['Close'], label="Historical", color='blue')
                ax.plot(forecast_series.index, forecast_series.values, label="Forecast", color='red')
                ax.set_title(f"{model_choice} Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)
            
            # Add download button for forecast data
            forecast_df = pd.DataFrame({"Forecast": forecast_series})
            csv = convert_df_to_csv(forecast_df)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name=f"{ticker}_{model_choice}_forecast.csv",
                mime='text/csv',
            )
    
    with tab3:
        st.header("Additional Analysis")
        if show_decomposition:
            st.subheader("Seasonal Decomposition")
            fig_dec = seasonal_decomposition_plot(data)
            st.pyplot(fig_dec)
        else:
            st.info("Enable 'Show seasonal decomposition' in the sidebar to view this analysis.")
        
        st.markdown("""
        **Tips:**
        - Use the interactive chart option for zoom and hover functionalities.
        - Download the forecast results for further analysis.
        - Adjust the model parameters in the sidebar to see different forecasting behaviors.
        """)
