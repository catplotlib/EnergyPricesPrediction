#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#setting figure size
plt.rcParams['figure.figsize'] = 20,10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

#read the file
data= pd.read_csv('8daysMin - PriceMinute.csv')

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))


#setting index
data.index = data.Date
data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = data.values
train = dataset[0:518,:]
valid = dataset[518:,:]

#converting dataset into x_train and y_train
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=11, batch_size=1, verbose=2)

#Using past 60 from the train data
inputs = data[len(data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

#root means square error values
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

#r2 scores
x = valid.reshape(-1,1)
error=r2_score(x, closing_price, sample_weight=None, multioutput='uniform_average')

#setting indexes for graphs
train = data[:518]
valid = data[518:]

#plotting the training data and new Predictions
valid['Predictions'] = closing_price
xmin, xmax = plt.xlim()
plt.plot(train['MCP'])
plt.plot(valid[['MCP', 'Predictions']])
plt.xlabel('Days (Data 11/07/2020 to 18/07/2020)')
plt.ylabel('MCP (Market Clearing Price) in Rupees/MWh')
plt.xlim([0,xmax])
plt.title('Weekly Prediction')
plt.xticks([0,96,192,288,384,480,576,672],['Saturday','Sunday','Monday',
                                            'Tuesday','Wednesday', 'Thursday',
                                            'Friday','Saturday'])

blue_patch = mpatches.Patch(color='#5497c5', label='Training Data(Price)')
orange_patch = mpatches.Patch(color='#ff902e', label='Validating Data(Price)')
green_patch = mpatches.Patch(color='#3ba73b', label='Prediction(Price)')
plt.legend(handles=[blue_patch,orange_patch,green_patch])                                     
plt.show() 
