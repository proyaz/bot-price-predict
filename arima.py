import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load the historical XRP price data into a pandas DataFrame
data = pd.read_csv("xrp.csv")

# Preprocess the data
data.dropna(inplace=True)  # Remove missing data
data["Date"] = pd.to_datetime(data["Date"])  # Convert date to datetime format
data.set_index("Date", inplace=True)  # Set date as index

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Train an ARIMA model
model = ARIMA(train_data["Adj Close"], order=(1, 1, 1))
model_fit = model.fit()

# Make predictions for the next 10 days
forecast = model_fit.forecast(steps=len(test_data))

# Format the output
dates = pd.date_range(
    start=test_data.index[0], periods=len(test_data), freq="D")
new_data = pd.DataFrame({"Adj Close": forecast}, index=dates)
output = pd.concat([test_data, new_data], axis=0)

print(output)
