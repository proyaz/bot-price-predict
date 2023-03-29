import pandas as pd
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# load data
data = pd.read_csv('xrp.csv')

# split data into train and test sets
train_data = data[:-10]
test_data = data[-10:]

# create input layer
inputs = Input(
    shape=(train_data[['Open', 'High', 'Low', 'Volume']].shape[1], 1))

# define convolutional layer
conv1 = Conv1D(filters=64, kernel_size=2,
               padding='causal', activation='relu')(inputs)
conv2 = Conv1D(filters=64, kernel_size=2,
               padding='causal', activation='relu')(conv1)

# define residual block
residual1 = Conv1D(filters=128, kernel_size=1, padding='valid')(conv2)
residual1 = Lambda(lambda x: x * 0.8)(residual1)
skip1 = Conv1D(filters=10, kernel_size=1, padding='valid')(conv2)
skip1 = Lambda(lambda x: x * 0.2)(skip1)
residual1 = Add()([conv2, residual1])
outputs1 = Add()([skip1, residual1])

# define residual block
residual2 = Conv1D(filters=128, kernel_size=1, padding='valid')(outputs1)
residual2 = Lambda(lambda x: x * 0.8)(residual2)
skip2 = Conv1D(filters=10, kernel_size=1, padding='valid')(outputs1)
skip2 = Lambda(lambda x: x * 0.2)(skip2)
residual2 = Add()([outputs1, residual2])
outputs2 = Add()([skip2, residual2])
