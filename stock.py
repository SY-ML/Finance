import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

# https://www.kaggle.com/general/272226
# https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the ticker symbol
tickerSymbol = 'MVST'

# Get data on this ticker
data = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
data = data.history(period='1d', start='2019-1-1', end='2023-12-31')

# Assuming 'data' is your DataFrame and 'High' is the target column
data_input = data.drop('High', axis=1)
data_target = data['High']

print(data_input)
print(data_target)
exit()
# Scale the input data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
data_input_scaled = scaler.fit_transform(data_input)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_input_scaled, data_target, test_size=0.2, random_state=42)

# Reshape input to be 3D [samples, timesteps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

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

import matplotlib.pyplot as plt

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
plt.figure(figsize=(10,6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(loc='upper right')
plt.show()