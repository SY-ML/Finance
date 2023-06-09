import datetime
import yfinance as yf
import os
import pandas as pd
# import tensorflow as tf
# print("GPU Available: ", tf.config.list_physical_devices('GPU'))
# print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
# print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

# https://www.kaggle.com/general/272226
# https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.preprocessing import MinMaxScaler

# 060923 출석체크 only :(
print(1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Lithium-Ion Battery Manufacturers:

Contemporary Amperex Technology Co., Ltd. (CATL): 300750 on the Shenzhen Stock Exchange​1​.
LG Chem, Ltd.: 051910 on the Korea Exchange​2​.
Panasonic Corporation: 6752 on the Tokyo Stock Exchange and PCRFY in OTC markets​3​​4​.
BYD Company Limited: BYDDY in OTC markets and 1211 on the Hong Kong Stock Exchange​5​​6​.
Tesla, Inc.: TSLA on the NASDAQ​7​.
Solid-State Battery Manufacturers:

Solid Power, Inc.: SLDP on the NASDAQ​8​.
QuantumScape Corporation: QS on the NYSE​9​.
"""

# 060923 출석 only :(
class YFStockPrice():
    def __init__(self, ticker, date_from, date_to, period='1d'):
        self._ticker = ticker
        self._date_from = date_from
        self._date_to = date_to
        self._period = period

        self._file_name = f'{ticker}({self._date_from.replace("-", "")}~{self._date_to.replace("-","")})_{period}).csv'

        self.data = self.load_data()

    def fetch_stock_price_via_yfinance(self):
        data = yf.Ticker(self._ticker)
        data = data.history(period=self._period, start=self._date_from, end=self._date_to)
        return data

    def load_data(self):
        file_path = os.path.join('./yfinance', self._file_name)

        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            data = self.fetch_stock_price_via_yfinance()
            data.to_csv(file_path, index=False)

        return data


dict_ticker = {
    'CATL': '300750.SZ',
    'LGChem': '051910.KS',
    'Panasonic': '6752.T',
    'BYD': 'BYDDY',
    'SolidPower': 'SLDP',
    'QuantumScape': 'QS',
    'Tesla': 'TSLA'
}

date_from = '2010-01-01'
date_to = datetime.datetime.now().strftime('%Y-%m-%d')


dict_data = dict_ticker.copy()

for com_name, com_ticker in dict_ticker.items():
    sp = YFStockPrice(ticker= com_ticker , date_from= date_from, date_to= date_to)
    dict_data[com_name] = sp.data


# Define the ticker symbol
tickerSymbol = 'MVST'

# Get data on this ticker
data = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
data = data.history(period='1d', start='2019-1-1', end='2023-12-31')

# Assuming 'data' is your DataFrame and 'High' is the target column
data_input = data.drop('High', axis=1)
data_target = data['High']


def process_dataset_for_tensorflow(dataset):
    # Scale the input data to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_input_scaled = scaler.fit_transform(dataset)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_input_scaled, data_target, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, timesteps, features] for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return (X_train, X_test, y_train, y_test)

def build_LSTM_model(dataset):
    X_train, X_test, y_train, y_test = dataset
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer=Adam())

    # Create callback for early stopping on validation loss. If the loss does not decrease in
    # two consecutive tries, stop training
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # Create model checkpoint
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Fit the LSTM model
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=72,
        verbose=0,
        validation_split=0.2,  # set validation split for early stopping
        callbacks=[early_stopping, checkpoint]
    )

    # Fit the LSTM model
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=72,
        verbose=0,
        validation_split=0.2,  # set validation split for early stopping
        callbacks=[early_stopping, checkpoint]
    )

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(loc='upper right')
    plt.show()
