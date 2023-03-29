import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# load data
data = pd.read_csv('xrp.csv')

# split data into train and test sets
train_data = data[:-10]
test_data = data[-10:]

# train the model
model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model.fit(train_data[['Open', 'High', 'Low', 'Volume']], train_data['Close'])

# make predictions
forecast = model.predict(test_data[['Open', 'High', 'Low', 'Volume']])
test_data['Predictions'] = forecast

# print test data with predictions
print(test_data)
