import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# load data
data = pd.read_csv('xrp.csv')

# split data into train and test sets
train_data = data[:-10]
test_data = data[-10:]

# train the model
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
model = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
model.fit(train_data[['Open', 'High', 'Low', 'Volume']], train_data['Close'])

# make predictions
forecast = model.predict(test_data[['Open', 'High', 'Low', 'Volume']])
test_data['Predictions'] = forecast

# print test data with predictions
print(test_data)
