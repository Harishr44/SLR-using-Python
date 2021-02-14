# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:10:38 2020

@author: Harish
"""

#y=salary
#x=years of experience(yex)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sald=pd.read_csv("Salary_Data.csv")

plt.hist(sald.yex)
plt.hist(sald.salary)
plt.boxplot(sald.yex) #more than 50% of the data lies within 2.5 to 8 years of experience  no outliers
plt.boxplot(sald.salary) # more than 50% of data lies within 55000 to 100000 no outlirs

plt.plot(sald.yex,sald.salary,"ro");plt.xlabel(yex);plt.ylabel(salary)
# it seem both are linear and follows normal distribution wrt to each other and good corelated

sald.salary.corr(sald.yex)
#r= 0.978 very good value as strongly corelated

import statsmodels.formula.api as smf
model1=smf.ols('salary~yex',data=sald).fit()
model1.params
#y=25792.2+9449.96(x)

model1.summary()
#pvalue=0.00< 0.05
#R sq=0.957 
#both the values are very good 
#as our R sq value is very high we proceed with this model1 only


model1.conf_int(0.05)
pred=model1.predict(sald)
sald["pred"]=pred

import matplotlib.pylab as plt
plt.scatter(x=sald.yex,y=sald.salary,color='red');plt.plot(sald.yex,pred,color='blue');plt.xlabel('yearsofexperience');plt.ylabel('salary')


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(sald.salary,pred))
rmse
#5592.04

resid=pred-sald.salary
sald["resid"]=resid

sal_resid=model1.resid_pearson
sal_resid

plt.plot(pred,model1.resid_pearson,"o");plt.axhline(y=0,color='blue');plt.xlabel("Predicted");plt.ylabel("Actual")

plt.plot(pred,sald.salary,"o");plt.xlabel("Predicted");plt.ylabel("Actual")
plt.scatter(x=pred,y=sald.salary);plt.xlabel("Predicted");plt.ylabel("Actual")
