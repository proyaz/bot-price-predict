import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('xrp.csv')

# Preprocessing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor())
])

# Perform walk-forward validation using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Define the hyperparameter search space for RandomizedSearchCV
param_dist = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [2, 4, 6, 8, 10],
    'model__learning_rate': [0.1, 0.05, 0.01, 0.005],
    'model__subsample': [0.5, 0.75, 1.0],
    'model__min_samples_leaf': [1, 3, 5, 10]
}

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=50, cv=tscv, n_jobs=-1, random_state=42)
random_search.fit(data[['Open', 'High', 'Low', 'Volume']], data['Close'])

# Get the best model
best_model = random_search.best_estimator_


def generate_future_data(initial_data, model, days):
    future_data = initial_data.copy()
    for _ in range(days):
        last_data_point = future_data.iloc[-1]
        next_open = last_data_point['Close']
        next_high = next_open * (1 + np.random.uniform(0, 0.02))
        next_low = next_open * (1 - np.random.uniform(0, 0.02))
        next_volume = last_data_point['Volume'] * np.random.uniform(0.9, 1.1)
        next_data_point = pd.DataFrame({'Open': [next_open], 'High': [
                                       next_high], 'Low': [next_low], 'Volume': [next_volume]})
        next_close = model.predict(next_data_point)
        next_data_point['Close'] = next_close
        future_data = future_data.append(next_data_point, ignore_index=True)
    return future_data.tail(days)


# Generate and predict for the next 30 days
future_data = generate_future_data(data, best_model, 30)
print(future_data)
