import math, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from pathlib import Path
current_path = Path.cwd()

plt.style.use('fivethirtyeight')

df = pd.read_csv(current_path / 'Data/std_daily.csv',delimiter=',')

data_name = 'Close'
NUMBER_OF_DATA_IN_TRAIN = 21

data = df.filter([data_name])
dataset = data.values

training_data_len = math.ceil(len(dataset) * 0.8)

x_train = []
y_train = []

for i in range(NUMBER_OF_DATA_IN_TRAIN, training_data_len):
    x_train.append(dataset[i-NUMBER_OF_DATA_IN_TRAIN:i, 0])
    y_train.append(dataset[i,0])

    if i <= NUMBER_OF_DATA_IN_TRAIN:
        print(x_train)
        print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train.reshape(-1,1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(15))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)


test_data = dataset[training_data_len - NUMBER_OF_DATA_IN_TRAIN: , :]

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(NUMBER_OF_DATA_IN_TRAIN, len(test_data)):
    x_test.append(test_data[i-NUMBER_OF_DATA_IN_TRAIN:i, 0])

x_test = np.array(x_test)

x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test.reshape(-1,1))

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions - y_test)**2)))

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('DATE', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train[data_name])
plt.plot(valid[[data_name, 'Predictions']])
plt.legend(['Train', 'Val','Predictions'], loc='lower right')
plt.show()