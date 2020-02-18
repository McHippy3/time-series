import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

#read in data using pandas
train_df = pd.read_csv('data/hourly_wages_data.csv')

#check data has been read in properly
train_df.head()

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['wage_per_hour'])

#check that the target variable has been removed
print(train_X.head())

#create a dataframe with only the target column
train_y = train_df[['wage_per_hour']]

#view dataframe
train_y.head()

#create model
model = keras.Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(layers.Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = keras.callbacks.EarlyStopping(patience=3)

#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
test_y_predictions = model.predict(test_X)