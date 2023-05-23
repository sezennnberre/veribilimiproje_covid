# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:09:01 2023

@author: HP
"""

import pandas as pd
import numpy as np

world = pd.read_csv("worldometer_data.csv")

world = world.fillna(0)

popu_totcase = world.iloc[:,2:4].values
popu_totcase = pd.DataFrame(data=popu_totcase, columns=["Population","TotalCase"])
TotalDeath = world.iloc[:,5:6].values
TotalDeath = pd.DataFrame(data = TotalDeath, columns=["TotalDeath"])
TotalReco = world.iloc[:,7:8].values                                  #tahmin edilecek veri 
TotalReco = pd.DataFrame(data= TotalReco,columns=["TotalRecovered"])
Activecase = world.iloc[:,9:10].values
Activecase = pd.DataFrame(data = Activecase, columns=["ActiveCases"])
Totaltest= world.iloc[:,13:14].values
Totaltest = pd.DataFrame(data= Totaltest,columns = ["TotalTest"])

firstcolumn = pd.concat([popu_totcase,TotalDeath],axis=1)
secondcolumn = pd.concat([Activecase,Totaltest],axis = 1)
son_veri = pd.concat([firstcolumn,secondcolumn],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(son_veri,TotalReco,test_size=0.33,random_state=0)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10,n_jobs=-1)
rf.fit(x_train, y_train)
print(rf.score(x_train,y_train))

y_pred = rf.predict(x_train)

# r^2 ile hata hesabı
from sklearn.metrics import r2_score
print("Random Forest R2 degeri")
print(r2_score(rf.predict(x_train),y_train))
#değerler her çalıştırmada farklılık gösterebilir neden bilmiyorum.

# RMSLE ile hata hesabı       squared false yapınca root oluyor.
from sklearn.metrics import mean_squared_log_error
print("Mean Squared Log Error degeri")
print(mean_squared_log_error(rf.predict(x_train),y_train,squared=False))

#RMSE ile hata hesabı
from sklearn.metrics import mean_squared_error
print("Mean Squared Error degeri")
print(mean_squared_error( rf.predict(x_train),y_train,squared=False))



#LinerRegressor kullanarak
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(x_train,y_train)
y_pred_l = l.predict(x_test)


# r^2 ile hata hesabı
from sklearn.metrics import r2_score
print("Linear Regression ile Random Forest R2 degeri")
print(r2_score(l.predict(x_train),y_train))
#değerler her çalıştırmada farklılık gösterebilir neden bilmiyorum.

'''
# RMSLE ile hata hesabı       squared false yapınca root oluyor.
from sklearn.metrics import mean_squared_log_error
print("Linear Regression ile Mean Squared Log Error degeri")
print(mean_squared_log_error(l.predict(x_train),y_train,squared=False))
'''


#RMSE ile hata hesabı
from sklearn.metrics import mean_squared_error
print("Linear Regression ile Mean Squared Error degeri")
print(mean_squared_error( l.predict(x_train),y_train,squared=False))

















