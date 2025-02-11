# Time Series Forecasting App

A powerful and interactive web application for forecasting time series data—specifically stock closing prices—using three robust models: **ARIMA**, **Prophet**, and **LSTM**. The app is built using Streamlit and leverages live data from Yahoo Finance, along with additional datasets available on Kaggle.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Time Series Forecasting App is designed to help users explore and forecast stock closing prices using three forecasting models:

- **ARIMA:** A classical time series model that is ideal for linear patterns.
- **Prophet:** Developed by Facebook, this model is robust to missing data and handles seasonality and trend changes efficiently.
- **LSTM:** A deep learning model implemented in TensorFlow, capable of capturing complex non-linear patterns in time series data.

Users can interact with the app via a user-friendly dashboard, configure model parameters, and view both static and interactive charts. Additional features include seasonal decomposition analysis and the ability to download forecast results as a CSV file.

## Features

- **Model Selection:** Choose between ARIMA, Prophet, and LSTM models.
- **Data Sourcing:** Fetch live stock data from Yahoo Finance.
- **Customizable Forecast Parameters:** Adjust forecasting periods and model-specific parameters.
- **Interactive Visualization:** Toggle between static (Matplotlib) and interactive (Plotly) charts.
- **User-Selectable Date Range:** Zoom in on historical data with user-defined date ranges.
- **Seasonal Decomposition:** View the trend, seasonal, and residual components of your time series.
- **Download Forecast:** Export forecast results as a CSV file.
- **Custom Theming:** Enhanced user interface with custom CSS styling for a polished look.

## Tech Stack

- **Python 3.8+**
- **Streamlit:** For building interactive web applications.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **yfinance:** For downloading historical stock data.
- **Matplotlib & Plotly:** For data visualization.
- **Statsmodels:** For ARIMA and seasonal decomposition.
- **Prophet:** For robust forecasting with trend and seasonality.
- **TensorFlow & Keras:** For building and loading LSTM models.
- **scikit-learn:** For data preprocessing.

## Data Source

- **Yahoo Finance:** The primary source for live stock data is [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library.
- **Kaggle Dataset (Optional):** Alternatively, you can experiment with datasets such as [Historical Stock Prices](https://www.kaggle.com/datasets/rohitsahoo/historical-stock-prices).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/time-series-forecasting-app.git
   cd time-series-forecasting-app
   ```

2. Create a Virtual Environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

4.  Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

- Run the App:
- In your project directory, start the Streamlit app with:

  ```bash
  streamlit run app.py
  ```

1. Interact with the Dashboard:

- Use the sidebar to select the ticker, date range, forecast period, and model-specific parameters.
- Adjust the visualization date range to zoom in on the desired window.
- View interactive or static charts depending on your selection.
- Access additional analysis such as seasonal decomposition.
- Download the forecasted results as a CSV file using the provided download button.

### Project Structure

time-series-forecasting-app/
│
├── app.py # Main Streamlit application file
├── lstm_model.h5 # Pre-trained LSTM model (if applicable)
├── scaler.pkl # Preprocessing scaler for LSTM model (if applicable)
├── requirements.txt # List of required Python packages
└── README.md # Project documentation (this file)

### Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please open an issue or submit a pull request. Follow these steps:

- Fork the repository.
- Create a new branch: git checkout -b feature/YourFeatureName
- Commit your changes: git commit -am 'Add some feature'
- Push to the branch: git push origin feature/YourFeatureName
- Open a pull request.
