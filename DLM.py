import pandas as pd
from pydlm import dlm, trend, dynamic, seasonality

# read data
df = pd.read_csv('xrp.csv', parse_dates=['Date'], index_col='Date')

# create dlm object
model = dlm(df['Close'])

# add components to the model
model = model + trend(degree=1, name='linear trend', w=1.0)
model = model + \
    dynamic(features=df[['Open', 'High', 'Low', 'Volume']],
            name='external regressor')
model = model + seasonality(period=7, name='weekly seasonality', w=1.0)

# fit the model
model.fit()

# make predictions for next 10 days
predict_days = 10
predict = model.predictN(N=predict_days)

# get the prediction results
result = pd.DataFrame({
    'Open': predict[0],
    'High': predict[1],
    'Low': predict[2],
    'Close': predict[3],
    'Adj Close': predict[3],
    'Volume': df['Volume'].iloc[-1]
}, index=pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=predict_days, freq='D'))

# print the result
print(result)