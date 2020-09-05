#import packages
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#to plot within notebook
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pylab import rcParams

#setting figure size
rcParams['figure.figsize'] = 20,10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

#read the file
df = pd.read_csv('8daysMin - PriceMinute.csv')

#sorting
data = df.sort_index(ascending=True, axis=0)

#splitting training and validation Data
train = data[:518]
valid = data[518:]


#selecting relevant columns
x_train = train.drop('MCP', axis=1)
y_train = train['MCP']
x_valid = valid.drop('MCP', axis=1)
y_valid = valid['MCP']

#Selecting and training the machine learning model
model = LinearRegression()
model.fit(x_train,y_train)

#Predicting the values
preds = model.predict(x_valid)

#root means square error values
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

#r2 scores
x=valid['MCP'].values
x = x.reshape(-1,1)
error=r2_score(x, preds, sample_weight=None, multioutput='uniform_average')


#setting indexes for graphs
valid['Predictions'] = preds
valid.index = data[518:].index
train.index = data[:518].index

#plotting the training data and new Predictions
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(train['MCP'])
plt.plot(valid[['MCP', 'Predictions']])
plt.xlabel('Days (Data from 11/07/2020 to 18/07/2020)')
plt.ylabel('MCP (Market Clearing Price) in Rupees/MWh')
plt.xlim([0,xmax])
plt.title('Weekly Prediction')
plt.xticks([0,100,200,300,400,500,600,700],['Saturday','Sunday','Monday',
                                            'Tuesday','Wednesday', 'Thursday',
                                            'Friday','Saturday'])
blue_patch = mpatches.Patch(color='#5497c5', label='Training Data(Price)')
orange_patch = mpatches.Patch(color='#ff902e', label='Validating Data(Price)')
green_patch = mpatches.Patch(color='#3ba73b', label='Prediction(Price)')
plt.legend(handles=[blue_patch,orange_patch,green_patch])                                         
plt.show()
