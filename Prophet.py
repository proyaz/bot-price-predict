import pandas as pd
import numpy as np
from fbprophet import Prophet

# Load the historical XRP price data into a pandas DataFrame
data = pd.read_csv("xrp.csv")

# Preprocess the data
data.dropna(inplace=True)  # Remove missing data
data["Date"] = pd.to_datetime(data["Date"])  # Convert date to datetime format
data.rename(columns={"Date": "ds", "Adj Close": "y"},
            inplace=True)  # Rename columns
data = data[["ds", "y"]]  # Select only relevant columns

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Train a Prophet model
model = Prophet()
model.fit(train_data)

# Make predictions for the next 10 days
future = model.make_future
