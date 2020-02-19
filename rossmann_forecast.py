import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Graph settings
mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['axes.grid'] = False

#Ensure Reproducibility
tf.random.set_seed(13)

#Reading data
df_sales = pd.read_csv('rossmann-data/train.csv', dtype={'StateHoliday':str, 'SchoolHoliday':str})

#Convert Date field from string to Datetime
df_sales['Date'] = pd.to_datetime(df_sales['Date'])

#Removing Values
df_sales = df_sales[df_sales['Open'] != "0"]
df_sales = df_sales.reset_index()

#groupby Date and sum the sales
df_sales = df_sales.groupby('Date').Sales.sum().reset_index()
print(df_sales.head(10))

df_sales.to_csv('df_sales.csv')
plt.title("Date vs Sales (All Stores)")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.scatter(df_sales['Date'], df_sales['Sales'])
plt.plot(df_sales['Date'], df_sales['Sales'])
plt.show()

#Create a new dataframe to model the difference
df_diff = df_sales.copy()

#Add previous sales to the next row
df_diff['Prev_Sales'] = df_diff['Sales'].shift(1)

#Drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['Sales'] - df_diff['Prev_Sales'])
df_diff.head(10)

#Create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['Prev_Sales'],axis=1)

#Adding lags
for inc in range(1,50):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)

#Drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

#Import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['Sales','Date'],axis=1)

#Split train and test set
train_set, test_set = df_model[0:-60].values, df_model[-60:].values

#Apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)

#Reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)

#Reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = keras.Sequential()
model.add(keras.layers.LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=150, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)

#Reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

#Rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))

#Reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

#Inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#Create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-61:].Date)
act_sales = list(df_sales[-61:].Sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['Date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)

#Merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='Date',how='left')

#Plot actual and predicted
plt.clf()
plt.title("Predicted vs Actual (All Stores)")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
actual_plot, = plt.plot(df_sales_pred['Date'][-100:], df_sales_pred['Sales'][-100:], '-o', c='b', label="Actual")
predicted_plot, = plt.plot(df_sales_pred['Date'][-100:], df_sales_pred['pred_value'][-100:], '-o', c='r', label="Predicted")
plt.legend(handles=[actual_plot, predicted_plot])
plt.show()

#Results
#loss: 0.0053