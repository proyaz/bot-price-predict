import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# Train and evaluate various time-series models
models = {
    #  "ARIMA": ARIMA(train_data["Adj Close"], order=(1, 1, 1)).fit(),
    #  "SARIMAX": SARIMAX(train_data["Adj Close"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 12)).fit(),
    #  "VAR": VAR(train_data.values).fit(maxlags=4),
    #   "VECM": VECM(train_data, k_ar_diff=1).fit(),
  #  "Holt-Winters": ExponentialSmoothing(train_data["Adj Close"], seasonal="add", seasonal_periods=12).fit(),
}

results = []
for name, model in models.items():
    if name == "VAR":
        forecast = model.forecast(model.y, steps=len(test_data))
    else:
        forecast = model.forecast(model.endog, steps=len(test_data))
    mse = mean_squared_error(test_data["Adj Close"], forecast)
    results.append({"Model": name, "MSE": mse})

# Print the results
print(pd.DataFrame(results))

# Choose the best model and make predictions for the next 10 days
best_model = models["SARIMAX"]
forecast = best_model.forecast(steps=10)

# Format the output
dates = pd.date_range(start=data.index[-1], periods=10+1, freq="D")[1:]
new_data = pd.DataFrame({"Adj Close": forecast}, index=dates)
output = pd.concat([data, new_data])

print(output.tail(10))
