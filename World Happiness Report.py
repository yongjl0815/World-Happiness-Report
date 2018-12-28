import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# to evaluate our model performance
from sklearn.metrics import mean_squared_error, r2_score

# algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# for data visualization
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings            
warnings.filterwarnings("ignore") 

from pandas.tools.plotting import parallel_coordinates

import os

# Read the data file
data = pd.read_csv('../input/2015.csv')

print ("Shape : ")
print (data.shape)

# print 10 random samples
data.sample(10)

# Check for missing data
print('Missing values in each column:\n\n', data.isnull().sum())

#Replace space in column names with _ 
data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.replace('(', '')
data.columns = data.columns.str.replace(')', '')

# Check info
data.info()


#===========================================Data Analysis========================================
# happiness ratio accrding t region
region_lists=list(data['Region'].unique())
region_happiness_ratio=[]
for each in region_lists:
    region=data[data['Region']==each]
    region_happiness_rate=sum(region.Happiness_Score)/len(region)
    region_happiness_ratio.append(region_happiness_rate)
    
data_temp=pd.DataFrame({'region':region_lists,'region_happiness_ratio':region_happiness_ratio})
new_index=(data_temp['region_happiness_ratio'].sort_values(ascending=False)).index.values
sorted_data = data_temp.reindex(new_index)

sorted_data

# bar chart
plt.figure(figsize=(12,10))
sns.barplot(x=sorted_data['region'], y=sorted_data['region_happiness_ratio'],palette=sns.cubehelix_palette(len(sorted_data['region'])))

plt.xticks(rotation= 90)
plt.xlabel('Region')
plt.ylabel('Region Happiness Ratio')
plt.title('Happiness rate for regions')
plt.show()

#Horizontal bar showing factors affecting happiness in each region
region_lists=list(data['Region'].unique())
share_economy=[]
share_family=[]
share_health=[]
share_freedom=[]
share_trust=[]
for each in region_lists:
    region=data[data['Region']==each]
    share_economy.append(sum(region.Economy_GDP_per_Capita)/len(region))
    share_family.append(sum(region.Family)/len(region))
    share_health.append(sum(region.Health_Life_Expectancy)/len(region))
    share_freedom.append(sum(region.Freedom)/len(region))
    share_trust.append(sum(region.Trust_Government_Corruption)/len(region))
#Visualization
f,ax = plt.subplots(figsize = (9,5))
sns.set_color_codes("pastel")
sns.barplot(x=share_economy,y=region_lists,color='g',label="Economy")
sns.barplot(x=share_family,y=region_lists,color='b',label="Family")
sns.barplot(x=share_health,y=region_lists,color='c',label="Health")
sns.barplot(x=share_freedom,y=region_lists,color='y',label="Freedom")
sns.barplot(x=share_trust,y=region_lists,color='r',label="Trust")
ax.legend(loc="lower right",frameon = True)
ax.set(xlabel='Percentage of Region', ylabel='Region',title = "Factors affecting happiness score")
plt.show()

# pair plot
pp = sns.pairplot(data=data, y_vars=['Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy', 'Freedom', 'Trust_Government_Corruption', 'Generosity', 'Dystopia_Residual', 'Region'], x_vars=['Happiness_Score'])
#===========================================Data Analysis========================================



# check unique values in Region column
#print (data.Region.unique())

#converting region names to numerical numbers
data['Region'] = data['Region'].map({'Western Europe' : 1, 'North America' : 2, 'Australia and New Zealand': 3, 'Middle East and Northern Africa': 4, 'Latin America and Caribbean' : 5, 'Southeastern Asia': 6, 'Central and Eastern Europe': 7, 'Eastern Asia' : 8,'Sub-Saharan Africa': 9, 'Southern Asia': 10})


# separate our target (y) from our input (X)
y = data.Happiness_Score
X = data.drop(['Country', 'Happiness_Score'], axis=1)
#x.info()

#split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# normalization
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}

# cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#clf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
clf.fit(X_train, y_train)
 
# Evaluate model on test data
pred = clf.predict(X_test)
print (r2_score(y_test, pred)) 
print (mean_squared_error(y_test, pred)) 

# accuracy
errors = abs(pred- y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#Linear Regression Model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
regression_model.score(X_test, y_test)
				  