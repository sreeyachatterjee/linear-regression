import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
ds=pd.read_csv('C:/Users/sreey/Downloads/SimpleLinearRegression.csv')
ds
x=ds.iloc[:,:-1].values
y=ds.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)
mtp.scatter(x_train,y_train,color="green")
mtp.plot(x_train,x_pred,color="red")
mtp.title("Salary vs experience (trainning dataset)")
mtp.xlabel("Years of experience")
mtp.ylabel("Salary(In Rupees)")
mtp.show()
#visualizing the result

mtp.scatter(x_test,y_test,color='blue')
mtp.plot(x_train,x_pred,color='red')
mtp.title("Salary vs Experience")
mtp.xlabel("Year Of Experience")
mtp.ylabel("Salary(In Rupees)")
mtp.show()