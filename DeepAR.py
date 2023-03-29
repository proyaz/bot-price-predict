import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

# load data
data = pd.read_csv('XRP-USD.csv')

# convert data to list of dicts
data_list = []
for i in range(len(data)):
    data_dict = {
        'start': data.iloc[i]['Date'],
        'target': data.iloc[i][['Open', 'High', 'Low', 'Close']].tolist()
    }
    data_list.append(data_dict)

# create list dataset
prediction_length = 10
freq = '1D'
custom_ds = ListDataset(data_list, freq=freq)

# define estimator
estimator = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    trainer=Trainer(ctx="cpu", epochs=10, learning_rate=1e-3, batch_size=32),
    num_cells=40,
    num_layers=2,
    dropout_rate=0.1,
    use_feat_dynamic_real=True,
    cardinality=[1],
    embedding_dimension=[10]
)

# train model
predictor = estimator.train(custom_ds)

# evaluate model
forecast_it, ts_it = make_evaluation_predictions(
    dataset=custom_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(
    iter(tss), iter(forecasts), num_series=len(custom_ds))

# print evaluation results
print(agg_metrics)
