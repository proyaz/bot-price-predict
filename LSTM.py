import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# load data
df = pd.read_csv('xrp.csv')

# preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# split data
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

# create dataset


def create_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data)-time_steps-1):
        X.append(data[i:(i+time_steps), 0])
        Y.append(data[(i+time_steps), 0])
    return np.array(X), np.array(Y)


time_steps = 5
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# train model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# calculate root mean squared error
train_error = np.sqrt(np.mean(np.power((Y_train - train_predict), 2)))
test_error = np.sqrt(np.mean(np.power((Y_test - test_predict), 2)))
print('Train RMSE:', train_error)
print('Test RMSE:', test_error)

# create dataframe with predictions
train_predict_df = pd.DataFrame(train_predict, columns=[
                                'Predictions'], index=df.index[:train_size][-len(train_predict):])
test_predict_df = pd.DataFrame(test_predict, columns=[
                               'Predictions'], index=df.index[train_size+time_steps+1:][-len(test_predict):])
predict_df = pd.concat([train_predict_df, test_predict_df])

# merge dataframe with original data
df = pd.concat([df, predict_df], axis=1)

# output predictions
print(df.tail(10))
