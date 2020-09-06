#import packages
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.svm import SVR
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
data = pd.read_csv('8daysMin - PriceMinute.csv')

#splitting training and validation Data

train = data[:518]
valid = data[518:]


#selecting relevant columns
x_train = train.drop('MCP', axis=1)
y_train = train['MCP']
x_valid = valid.drop('MCP', axis=1)
y_valid = valid['MCP']

#Selecting and training the machine learning model
model = SVR(kernel= 'rbf', C= 1e3, gamma= 'scale') #A lower C will encourage a larger margin, therefore a simpler decision function.
model.fit(x_train,y_train)

#Predicting the values
preds = model.predict(x_valid)

#root means square error values
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

#r2 scores
x=valid['MCP'].values
error=r2_score(x, preds, sample_weight=None, multioutput='uniform_average')


#setting indexes for graphs
valid['Predictions'] = preds
valid.index = data[518:].index
train.index = data[:518].index

#plotting the training data and new Predictions
xmin, xmax = plt.xlim()
plt.plot(train['MCP'])
plt.plot(valid[['MCP', 'Predictions']])
plt.xlabel('Days (Data from 11/07/2020 to 18/07/2020)')
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
