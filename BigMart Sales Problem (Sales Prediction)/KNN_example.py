#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
path = "c/Users/Bahador/Desktop/GoogleDrive/Different machine learaning Techniques"
df_train_raw = pd.read_csv('train.csv')
df_test_raw = pd.read_csv('test.csv')
#%%
output_tag = 'Item_Outlet_Sales'
print('df_train_raw.shape', df_train_raw.shape)
print(df_train_raw.nunique())

## Is ther any duplicate row in the training dataset?
dummy1 = df_train_raw.drop_duplicates()
dummy2 = df_test_raw.drop_duplicates()
print('Train dataset does not have any duplicate rows') if dummy1.shape[0] == df_train_raw.shape[0] else print('Train dataset HAS duplicate rows') 
print('Test dataset does not have any duplicate rows') if dummy2.shape[0] == df_test_raw.shape[0] else  print('Test dataset HAS duplicate rows') 
del dummy1, dummy2
#%%
'''
a Tiny EDA and pre-processing the Training dataset
'''
## NAN and Missing Value handeling
print(df_train_raw.isnull().sum())
# Then we try to impute the null values properly
df_train_raw['Item_Weight'].fillna(df_train_raw['Item_Weight'].mean(), inplace=True)
dummy_mode = df_train_raw['Outlet_Size'].mode()
df_train_raw['Outlet_Size'].fillna(dummy_mode[0], inplace=True)
del dummy_mode

'''
a Tiny EDA and pre-processing the Test dataset
'''
## NAN and Missing Value handeling
print(df_test_raw.isnull().sum())
# Then we try to impute the null values properly
df_test_raw['Item_Weight'].fillna(df_test_raw['Item_Weight'].mean(), inplace=True)
dummy_mode = df_test_raw['Outlet_Size'].mode()
df_test_raw['Outlet_Size'].fillna(dummy_mode[0], inplace=True)
del dummy_mode
#%%
'''
Since one of the categorical variables had a different duplicates in terms of the category we have to identify and remove them
'''
print(df_train_raw['Item_Fat_Content'].unique())
df_train_raw['Item_Fat_Content'].replace(['LF', 'low fat', 'reg'], ['Low Fat', 'Low Fat' ,'Regular' ], inplace=True)
sns.countplot('Item_Fat_Content',data=df_train_raw)
## We can repeat using this sns.coutplot to acquire more knowledge about the types and the sales



#%%
'''
Here I would like to do a correlation analysis between some of the
'''
## Be advised that the .corr() function on dataframe will automatically ignore the categorical columns
sns.heatmap(df_train_raw.corr(), annot=True, cmap='GnBu')
## By looking at the result, we can understand that hte item_MRP has the maximum correlation with the sales.
#%% 
'''
Pre processing and encoding the categorical features
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print(df_train_raw.columns)
categorical_columns = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']


for col in categorical_columns:
    df_train_raw[col] = le.fit_transform(df_train_raw[col])
for col in categorical_columns:
    df_test_raw[col] = le.fit_transform(df_test_raw[col])

df_train_raw.head()
df_test_raw.head()

# %%
'''
According to our domain knowdege, we know that the outlet establishment date is irrelevent
to the goal of this practice, hence should be removed.

Also, "Item_Identifier" and "Outlet_Identifier" should be remvoed if we just want to keep those impactfull features.
'''
df_train = df_train_raw.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)
df_test = df_test_raw.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)


## Nowe we build or X and Y for training
X = df_train.drop(['Item_Outlet_Sales'], axis=1)
Y = df_train['Item_Outlet_Sales']

from sklearn.model_selection import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 0)
#%%
'''
Here we perform some wellknown approach on regression analysis
'''
## Linear Regression
from sklearn.linear_model import LinearRegression
Linear_reg = LinearRegression()
Linear_reg.fit(X_train, Y_train)
Y_pred = Linear_reg.predict(X_test)

print('Train R2 =>>',Linear_reg.score(X_train, Y_train))
print('Test R2 =>>',Linear_reg.score(X_test, Y_test))

#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(X_train, Y_train)

Y_pred_dt = dt.predict(X_test)

print('Decision Tree - Train R2 =>>',dt.score(X_train, Y_train))
print('Decision Tree - Test R2 =>>',dt.score(X_test, Y_test))

#%% Random Forest REgressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

print('Random Forest - Train R2 =>>',rf.score(X_train, Y_train))
print('Random Forest - Test R2 =>>',rf.score(X_test, Y_test))

#%% SVM Regression
from sklearn.svm import SVR

svr = SVR(kernel='rbf', epsilon=0.1)
svr.fit(X_train, Y_train)
Y_pred_svr = svr.predict(X_test)

print('SVR - Train R2 =>>',svr.score(X_train, Y_train))
print('SVR - Test R2 =>>',svr.score(X_test, Y_test))

#%% KNN Regression
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import math

### Here I manully tries to find the best K using elbow curve
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = math.sqrt(mean_squared_error(Y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

### Here we use gride search algorithm to automatically find the best answer
from sklearn.model_selection import GridSearchCV

