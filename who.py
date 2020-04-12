# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:46:16 2020

@author: kshelar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

missing_values = [" ", "nan","", None, "NaN"]

data = pd.read_csv("Life Expectancy Data.csv", na_values = missing_values)

print(len(data['Country'].unique()))
print(len(data['Status'].unique())) #convert into categorical variable

mean_life = data['Life expectancy '].mean()
data['Life expectancy '].fillna(mean_life, inplace=True)

mean_Adult_Mortality = data['Adult Mortality'].mean()
data['Adult Mortality'].fillna(mean_Adult_Mortality, inplace=True)
print(mean_Adult_Mortality)

mean_Life_expectancy  = data['Life expectancy '].mean()
data['Life expectancy '].fillna(mean_Life_expectancy, inplace=True)

mean_Alcohol  = data['Alcohol'].mean()
data['Alcohol'].fillna(mean_Alcohol, inplace=True)

mean_Hepatitis  = data['Hepatitis B'].mean()
data['Hepatitis B'].fillna(mean_Hepatitis, inplace=True)

mean_Bmi  = data[' BMI '].mean()
data[' BMI '].fillna(mean_Bmi, inplace=True)

mean_Polio  = data['Polio'].mean()
data['Polio'].fillna(mean_Polio, inplace=True)

mean_totExpend  = data['Total expenditure'].mean()
data['Total expenditure'].fillna(mean_totExpend, inplace=True)

mean_dip  = data['Diphtheria '].mean()
data['Diphtheria '].fillna(mean_dip, inplace=True)

mean_gdp  = data['GDP'].mean()
data['GDP'].fillna(mean_gdp, inplace=True)

mean_pop = data['Population'].mean()
data['Population'].fillna(mean_pop, inplace=True)

mean_tnY1 = data[' thinness  1-19 years'].mean()
data[' thinness  1-19 years'].fillna(mean_tnY1, inplace=True)

mean_tnY2 = data[' thinness 5-9 years'].mean()
data[' thinness 5-9 years'].fillna(mean_tnY2, inplace=True)

mean_income = data['Income composition of resources'].mean()
data['Income composition of resources'].fillna(mean_income, inplace=True)

mean_school = data['Schooling'].mean()
data['Schooling'].fillna(mean_school, inplace=True)

#sns.pairplot(data)
Y = data['Life expectancy ']

X = data.drop('Country', axis=1).drop('Life expectancy ', axis=1)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("one_hot_encoder",OneHotEncoder(),[1])],remainder='passthrough')	
X = ct.fit_transform(X)
X = X[:,1:]


from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

prediction = lm.predict(X_test)

plt.scatter(y_test,prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#Backward Elimination

import statsmodels.formula.api as sm

X =np.append(arr = np.ones((2938,1)).astype(int), values = X, axis = 1) #2937 19 

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()

regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20]]

regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,2,3,4,5,7,8,9,10,11,12,13,14,15,17,19,20]]

regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5,7,8,9,10,11,12,13,14,15,17,19,20]]

regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5,7,9,10,11,12,13,14,15,17,19,20]]

regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()
'''
X_train, X_test, y_train, y_test = train_test_split(X_opt, Y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

prediction = lm.predict(X_test)

plt.scatter(y_test,prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y_test, prediction)))'''
