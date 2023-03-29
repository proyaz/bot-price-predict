import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Train a SARIMAX model
model = SARIMAX(train_data["Adj Close"], order=(
    1, 1, 1), seasonal_order=(0, 0, 0, 12))
model_fit = model.fit()

# Make predictions for the next 10 days
forecast = model_fit.forecast(steps=10)

# Format the output
dates = pd.date_range(start=data.index[-1], periods=10+1, freq="D")[1:]
new_data = pd.DataFrame({"Adj Close": forecast}, index=dates)
output = pd.concat([data, new_data])

print(output.tail(10))
