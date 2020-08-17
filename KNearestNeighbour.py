#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pylab import rcParams

#setting figure size
rcParams['figure.figsize'] = 20,10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22 

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('8daysMin - PriceMinute.csv')

#setting index as date
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#split into train and validation
train = data[:518]
valid = data[518:]

#selecting relevant columns
x_train = train.drop('MCP', axis=1)
y_train = train['MCP']
x_valid = valid.drop('MCP', axis=1)
y_valid = valid['MCP']

#scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#root means square error values
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

#r2 scores
x=valid['MCP'].values
x = x.reshape(-1,1)
from sklearn.metrics import r2_score
error=r2_score(x, preds, sample_weight=None, multioutput='uniform_average')

#setting indexes for graphs
valid['Predictions'] = preds
valid.index = data[518:].index
train.index = data[:518].index

#plotting the training data and new Predictions
plt.figure(figsize = (20,10))
plt.plot(train['MCP'])
plt.plot(valid[['MCP', 'Predictions']])
plt.xlabel('Days (Data from 11/07/2020 to 18/07/2020)')
plt.ylabel('MCP (Market Clearing Price) in Rupees/MWh')
plt.title('Weekly Prediction')
plt.xticks([0,100,200,300,400,500,600,700],['Saturday','Sunday','Monday',
                                            'Tuesday','Wednesday', 'Thursday',
                                            'Friday','Saturday'])
blue_patch = mpatches.Patch(color='#5497c5', label='Training Data(Price)')
orange_patch = mpatches.Patch(color='#ff902e', label='Validating Data(Price)')
green_patch = mpatches.Patch(color='#3ba73b', label='Prediction(Price)')
plt.legend(handles=[blue_patch,orange_patch,green_patch])                                    
plt.show() 

