import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#Setting constants
TRAIN_SPLIT = 700

tf.random.set_seed(13)

BATCH_SIZE = 256
BUFFER_SIZE = 100

EVALUATION_INTERVAL = 780
EPOCHS = 3

#Reading data
df_sales = pd.read_csv('rossmann-data/train.csv', dtype={'StateHoliday':str, 'SchoolHoliday':str})

#Only using Store 1 Sales
df_sales = df_sales[df_sales.Store == 1.0]
df_sales['Date'] = pd.to_datetime(df_sales['Date'].iloc[0:])
df_sales.set_index('Date')

#Taking out sales of zero
indexNames = df_sales[df_sales['Sales'] == 0].index
df_sales.drop(indexNames, inplace=True)
df_sales = df_sales.reset_index()

plt.scatter(df_sales['Date'], df_sales['Sales'])
plt.plot(df_sales['Date'], df_sales['Sales'])
#plt.show()

#Creating feature set
features_considered = ['DayOfWeek', 'Sales','Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
features = df_sales[features_considered]
features.index = df_sales['Date']

#Standardize sales and customer
dataset = features[features_considered]

sales_mean = dataset.Sales[:TRAIN_SPLIT].mean(axis=0)
sales_std = dataset.Sales[:TRAIN_SPLIT].std(axis=0)
dataset['Sales'] = (dataset['Sales']-sales_mean)/sales_std

customer_mean = dataset.Customers[:TRAIN_SPLIT].mean(axis=0)
customer_std = dataset.Customers[:TRAIN_SPLIT].std(axis=0)
dataset['Customers'] = (dataset['Customers']-customer_mean)/customer_std
dataset = dataset.values

#Splitting data
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
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

past_history = 640
future_target = 60
STEP = 5

# Multistep prediction
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

def create_time_steps(length):
  return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

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

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])