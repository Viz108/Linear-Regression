import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from statsmodels.stats.anova import anova_lm 

data = pd.read_csv("Boston.csv")
print(data.columns)

#create the predictor matrix
#only one predictor first, lstat
#need to include a column for the intercept value

X = pd.DataFrame({'intercept': np.ones(data.shape[0]), 'lstat': data['lstat']})

#create the response matrix
#looking for the medv variable

y = data['medv']

#fit to linear regression model 
#using the statsmodel function instead of sklearn

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
print(results.params)

#make prediction input matrix

predictX = pd.DataFrame({'intercept': np.ones(3), 'lstat':[5,10,15]})
print(predictX)

#make prediction and print mean and confidence intervals

predictions = results.get_prediction(predictX)
#mean
print(predictions.predicted_mean, '\n')
#confidence interval
print(predictions.conf_int(alpha=0.5), '\n')
#prediction interval 
print(predictions.conf_int(alpha=0.5, obs=True), '\n')

#plot results

#first, plot test data
plot = data.plot.scatter('lstat', 'medv')
plot.plot()
plt.show()

