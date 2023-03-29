import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# load data
data = pd.read_csv('xrp.csv')

# split data into train and test sets
train_data = data[:-15]
test_data = data[-15:]

# create a pipeline
scaler = StandardScaler()
selector = SelectKBest(score_func=f_regression, k='all')
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, alpha=0.1)

pipeline = Pipeline(
    [('scaler', scaler), ('selector', selector), ('model', model)])

# perform walk-forward validation
# tscv = TimeSeriesSplit(n_splits=5)
tscv = TimeSeriesSplit(n_splits=10)


# perform hyperparameter tuning using GridSearchCV
# param_grid = {
#    'model__n_estimators': [50, 100, 200],
#    'model__max_depth': [3, 5, 7],
#    'model__alpha': [0.1, 0.01]
# }
param_grid = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [2, 4, 6, 8, 10],
    'model__learning_rate': [0.1, 0.05, 0.01, 0.005],
    'model__subsample': [0.5, 0.75, 1.0],
    'model__min_samples_leaf': [1, 3, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, n_jobs=-1)
grid_search.fit(
    train_data[['Open', 'High', 'Low', 'Volume']], train_data['Close'])

best_model = grid_search.best_estimator_

# function to generate future data points


def generate_future_data(last_data_point, model, num_days):
    future_data = []
    current_data = last_data_point

    for _ in range(num_days):
        input_data_dict = {
            'Open': [current_data['Open']],
            'High': [current_data['High']],
            'Low': [current_data['Low']],
            'Volume': [current_data['Volume']],
        }
        input_data = pd.DataFrame(input_data_dict)
        prediction = model.predict(input_data)[0]

        next_data_point = {
            'Open': current_data['Close'],
            'High': max(current_data['Close'], prediction),
            'Low': min(current_data['Close'], prediction),
            'Close': prediction,
            # Use the previous day's volume or implement your own logic
            'Volume': current_data['Volume']
        }

        future_data.append(next_data_point)
        current_data = next_data_point

    return future_data


# generate 30 days of future data points
last_data_point = data.iloc[-1]
future_data = generate_future_data(last_data_point, best_model, 30)

# predict for the next 30 days
future_df = pd.DataFrame(future_data)
future_predictions = best_model.predict(
    future_df[['Open', 'High', 'Low', 'Volume']])
future_df['Predictions'] = future_predictions

# print future predictions
print(future_df)
