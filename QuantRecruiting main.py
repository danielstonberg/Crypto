# Install necessary libraries
!pip3 install coinbase-advanced-py
!pip install coinbase pandas statsmodels
!pip install gspread
!pip install nbconvert

from coinbase.wallet.error import CoinbaseError

import pandas as pd
import uuid
import json
from datetime import datetime, timedelta
import requests
import time
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np


# Mount Google Drive
from google.colab import drive
import os

drive.mount('/content/drive')

# Install nbconvert
!pip install nbconvert

# Convert the Jupyter Notebook to a Python script
!jupyter nbconvert --to script /content/drive/MyDrive/QuantRecruiting/TradingBot.ipynb

# Define the old and new file paths
old_file_path = '/content/drive/MyDrive/QuantRecruiting/TradingBot.txt'
new_file_path = '/content/drive/MyDrive/QuantRecruiting/main.py'

# Rename the file
os.rename(old_file_path, new_file_path)


import gspread
from google.oauth2.service_account import Credentials
from google.colab import drive

# service account email: danny-125@cryptotrades-429213.iam.gserviceaccount.com
# service account id: 102642234583282890448
# serive account key: ed9181ec7f49b540d1d9d775b8543c952ab0c51f


# Path to your service account key file
SERVICE_ACCOUNT_FILE = '/content/drive/MyDrive/QuantRecruiting/GoogleServiceKey.json'

# Define the scope
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using the service account
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Authenticate and initialize the gspread client
gc = gspread.authorize(credentials)

# Open the Google Sheets document by its name or URL
spreadsheet = gc.open("CryptoTrading")

from coinbase.wallet.client import Client
from coinbase.rest import RESTClient
from json import dumps
import statsmodels.api as sm
from google.colab import userdata

# API key
api_key = userdata.get('CoinbaseApi')

# API secret (private key in this case)
api_secret = userdata.get('CoinbaseSecret')
# Initialize the Coinbase client
client = RESTClient(api_key=api_key, api_secret=api_secret)

def place_order_and_log_to_sheets(client, spreadsheet, product_id, quote_size, strategy_id="0"):
    # Generate a unique client order ID
    client_order_id = str(uuid.uuid4())

    # Place a market order
    try:
        order = client.market_order_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            quote_size=quote_size
        )
        print("Order response:", json.dumps(order, indent=2))  # Print the order response
    except Exception as e:
        print(f"Error placing order: {e}")
        return

    # Extract the success response
    success_response = order.get("success_response", {})
    order_id = success_response.get("order_id", None)
    product_id = success_response.get("product_id", "N/A")
    side = success_response.get("side", "N/A")

    if not order_id:
        print("Order ID not found in the success response")
        return

     # Get fills for the order with a retry mechanism
    fills = []
    for _ in range(5):  # Retry up to 5 times
        try:
            fills_response = client.get_fills(order_id=order_id)
            print("Fills response:", json.dumps(fills_response, indent=2))  # Print the fills response
            fills = fills_response.get("fills", [])
            if fills:
                break  # Exit the loop if fills are found
        except Exception as e:
            print(f"Error getting fills: {e}")
        time.sleep(1)  # Wait for 1 second before retrying
        print("Retrying...")

    if not fills:
        print("No fills found for the order")
        fills = [{}]  # Ensure we have at least one empty fill to create the DataFrame

    # Get the current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create a DataFrame for the order
    order_data = {
        "client_order_id": [client_order_id],
        "product_id": [product_id],
        "side": [side],
        "quote_size": [float(quote_size)],
        "created_at": [current_time]
    }
    order_df = pd.DataFrame(order_data)

    # Create a DataFrame for the fills
    fills_df = pd.DataFrame(fills)
    if not fills_df.empty:
      numeric_columns = ['size', 'price', 'commision']
      for col in numeric_columns:
        if col in fills_df.columns:
            fills_df[col] = pd.to_numeric(fills_df[col], errors='coerce')
      fills_df['holding_value'] = fills_df['size'] / fills_df['price']



    # Convert DataFrames to list of lists
    order_values = order_df.values.tolist()
    fills_values = fills_df.values.tolist()


    # Select the "Orders" worksheet and append the order data
    orders_worksheet = spreadsheet.worksheet("Orders")
    orders_worksheet.append_rows(order_values, value_input_option='USER_ENTERED')

    # Select the "Fills" worksheet and append the fills data
    fills_worksheet = spreadsheet.worksheet("Fills")
    fills_worksheet.append_rows(fills_values, value_input_option='USER_ENTERED')

    print("Order data appended to Google Sheets 'Orders' sheet")
    print("Fills data appended to Google Sheets 'Fills' sheet")

CRYPTOCOMPARE_API_KEY = userdata.get('CryptoCompare')

def fetch_historical_data(asset, limit=365):
    url = f'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': asset,
        'tsym': 'USD',
        'limit': limit,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']].rename(columns={'time': 'date', 'close': 'price'})

# Define the time range for historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last 1 year

assets = ['LTC']

# Fetch historical data for each asset
historical_data = {}
for asset in assets:
    historical_data[asset] = fetch_historical_data(asset)
    historical_data[asset].set_index('date', inplace=True)

def fetch_live_price(client, currency_pair='LTC-USD'):
    try:
        url = f'https://api.coinbase.com/v2/prices/{currency_pair}/spot'
        response = requests.get(url)
        data = response.json()
        price = data['data']['amount']
        return float(price)
    except CoinbaseError as e:
        logging.error(f'Error fetching live price: {e}')
        return None

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_and_predict(data, seq_length=15, epochs=15):
    data = data.reshape(-1, 1)

    # Scale data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create training and test datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=epochs, validation_split=0.2)

    # Predicting
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return model, scaler, seq_length

def generate_trading_signals(actual_prices, predictions, buyThreshold=0.01, sellThreshold=0.05, period=5):
    signals = []
    for i in range(period, len(actual_prices)):
        if (actual_prices[i] > actual_prices[i-period] * (1 + buyThreshold)) and (predictions[i-period] < predictions[i]):
            signals.append('buy')
        elif actual_prices[i] < actual_prices[i-period] * (1 - sellThreshold):
            signals.append('sell')
        else:
            signals.append('hold')
    return signals

def run_trading_bot(client, model, scaler, seq_length, interval=1800):

    # Configure logging to output to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    prices = []

    while True:
        try:
            live_price = fetch_live_price(client, 'LTC-USD')
            if live_price is not None:
                prices.append(live_price)

                if len(prices) > seq_length:
                    scaled_prices = scaler.transform(np.array(prices[-seq_length:]).reshape(-1, 1))
                    X = np.array([scaled_prices])
                    prediction = model.predict(X)
                    prediction = scaler.inverse_transform(prediction)[0, 0]

                    actual_prices = np.array(prices[-seq_length:])
                    signals = generate_trading_signals(actual_prices, [prediction] * len(actual_prices))

                    if signals[-1] != 'hold':
                        quote_size = '5'  # Example value, adjust as needed
                        place_order_and_log_to_sheets(client, spreadsheet, 'LTC-USD', quote_size, signals[-1])

                    # Log current price and signal
                    logging.info(f'Price: {live_price}, Prediction: {prediction}, Signal: {signals[-1]}')

            time.sleep(interval)

        except Exception as e:
            logging.error(f'Error in trading bot: {e}')
            time.sleep(interval)

def main():
    # Assuming historical_data contains price data for LTC
    ltc_data = historical_data['LTC']['price'].values

    # Train LSTM model on LTC and generate predictions
    model, scaler, seq_length = train_and_predict(ltc_data)

    # Start the automated trading bot
    run_trading_bot(client, model, scaler, seq_length)

if __name__ == "__main__":
    main()
