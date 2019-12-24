from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

TRAIN_SPLIT = 300000

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# The weather dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='/Users/raymond/dev/tensorflow-time-series-forecasting/data/jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
# print(df.head())

# Part 2: Forecast a multivariate time series
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]
features.index = df['Date Time']
# print(features.head())

# features.plot(subplots=True)
# plt.show()

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# Multi-Step model

def multivariate_data(
  dataset, 
  target, 
  start_index, 
  end_index, 
  history_size,
  target_size, 
  step, 
  single_step=False
):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

future_target = 72
past_history = 720
future_target = 72
STEP = 6
BATCH_SIZE = 256
BUFFER_SIZE = 10000

x_train_multi, y_train_multi = multivariate_data(
  dataset, 
  dataset[:, 1], 
  0,
  TRAIN_SPLIT, past_history,
  future_target, 
  STEP
)
x_val_multi, y_val_multi = multivariate_data(
  dataset, 
  dataset[:, 1],
  TRAIN_SPLIT, 
  None,
  past_history,
  future_target, 
  STEP
)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(
    np.arange(num_out)/STEP,
    np.array(true_future), 
    'bo',
    label='True Future'
  )
  if prediction.any():
    plt.plot(
      np.arange(num_out)/STEP, 
      np.array(prediction), 
      'ro',
      label='Predicted Future'
    )
  plt.legend(loc='upper left')
  plt.show()

# for x, y in train_data_multi.take(1):
#   multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(
  32,
  return_sequences=True,
  input_shape=x_train_multi.shape[-2:])
)
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# for x, y in val_data_multi.take(1):
#   print (multi_step_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

multi_step_history = multi_step_model.fit(
  train_data_multi, 
  epochs=EPOCHS,
  steps_per_epoch=EVALUATION_INTERVAL,
  validation_data=val_data_multi,
  validation_steps=50
)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

# plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])