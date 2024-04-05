# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.
2.Compute predicted values.
3.Compute gradient of loss function.
4.Update weights using gradient descent.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sandeep V
RegisterNumber:  212223040179
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)
        
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:/Users/Aadhi/Documents/Software Engineering/50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
## Output:
```
![Screenshot 2024-03-05 094622](https://github.com/Dharma23012432/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/152275002/cb325bd7-1cd2-47f0-ad70-099aa1d9d1e1)
![WhatsApp Image 2024-03-05 at 10 20 02_2f41ae0c](https://github.com/Dharma23012432/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/152275002/0e7e4b2a-b953-44fb-809b-59c10d71cfff)
![image](https://github.com/Dharma23012432/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/152275002/9451f22f-4c80-4d37-8cd3-8e6f2ae09c01)
```




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
