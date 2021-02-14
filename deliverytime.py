# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:17:48 2020

@author: Harish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dt=pd.read_csv("delivery_time.csv")
plt.hist(dt.DeliveryTime)
plt.boxplot(dt.sortingtime)
plt.boxplot(dt.sortingTime)
plt.boxplot(dt.SortingTime)
plt.hist(dt.SortingTime)
plt.boxplot(dt.DeliveryTime)

plt.plot(dt.DeliveryTime,dt.SortingTime,"ro");plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')

dt.DeliveryTime.corr(dt.SortingTime)
np.corrcoef(dt.DeliveryTime,dt.SortingTime)
import statsmodels.formula.api as smf
model=smf.ols("DeliveryTime~SortingTime",data=dt).fit()
model.params
#y=6.58+1.649(x)

model.summary()
#pvalu=0.00<0.05
#rsq=0.68 which not so good

model.conf_int(0.05)
pred=model.predict(dt)
dt["pred"]=pred
pred.corr(dt.DeliveryTime)
#0.825
import matplotlib.pylab as plt
plt.scatter(x=dt.DeliveryTime,y=dt.SortingTime,color='blue');plt.plot(dt.SortingTime,dt.pred,color='black');plt.xlabel('SortingTime');plt.plot('DeliveryTime')

plt.scatter(x=dt.DeliveryTime,y=dt.SortingTime,color='blue')
plt.plot(dt.SortingTime,dt.pred,color='black')

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(dt.DeliveryTime,pred))
rmse
#rmse=2.79

#Log transformation
model2=smf.ols('DeliveryTime~np.log(SortingTime)',data=dt).fit()
model2.params
#y=1.159+9.04(x)

model2.summary()
#pvalue=0.000<0.05
#rsq=0.695 which is better than previous model

model.conf_int(0.05)
pred2=model2.predict(dt)
dt["pred2"]=pred2
pred2.corr(dt.DeliveryTime)
#0.833
plt.scatter(x=dt.DeliveryTime,y=dt.SortingTime,color='blue');plt.plot(dt.SortingTime,pred2,color='black');plt.xlabel('SortingTime');plt.plot('DeliveryTime')

rmse=sqrt(mean_squared_error(dt.DeliveryTime,pred2))
rmse
#rmse=2.733 which is better than previous model


#Exponential Transformation
model3=smf.ols('np.log(DeliveryTime)~SortingTime',data=dt).fit()
model3.params
#y=2.121+0.10(x)
model3.summary()
#pvalue=0.0<0.05
#rsq=0.711 which better than previous models
pred_lod=model3.predict(dt)
pred3=np.exp(pred_lod)
pred3.corr(dt.DeliveryTime)
#0.81
dt["pred3"]=pred3

plt.scatter(x=dt.DeliveryTime,y=dt.SortingTime,color='blue');plt.plot(dt.SortingTime,pred3,color='black');plt.xlabel('SortingTime');plt.plot('DeliveryTime')

rmse=sqrt(mean_squared_error(dt.DeliveryTime,pred3))
#2.94

#as R sq value is better for exponantial trans hence we go with model3
resid3=pred3-dt.DeliveryTime
resid_dt=model3.resid_pearson
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
plt.scatter(x=pred3,y=dt.DeliveryTime);plt.xlabel("Predicted");plt.ylabel("Actual")


#Quadratic model

dt["SortingTime_sq"]=dt.SortingTime*dt.SortingTime
model4=smf.ols("DeliveryTime~SortingTime+SortingTime_sq",data=dt).fit()
model4.params
model4.summary()
#pvalue=0.07>0.05
#not significant value hence rejected