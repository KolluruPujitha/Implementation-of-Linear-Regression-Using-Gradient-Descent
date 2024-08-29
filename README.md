# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1.Start

STEP 2.Intialize weights randomly.

STEP 3.Compute predicted.

STEP 4.Compute gradient of loss function.

STEP 5.Update weights using gradient descent.

STEP 6.End
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KOLLURU PUJITHA
RegisterNumber: 212223240074 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta usinng gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Desktop/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values) 
X1=X.astype(float)

scaler=StandardScaler()

y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)

Y1_Scaled=scaler.fit_transform(y)

print(X)

print(X1_Scaled)
```

## Output:
![Screenshot 2024-08-29 183418](https://github.com/user-attachments/assets/fef15727-051f-4449-85f5-3f5012d8716c)

![Screenshot 2024-08-29 182719](https://github.com/user-attachments/assets/8ef9b241-20a8-4cda-b79b-5aadab667f6f)

![Screenshot 2024-08-29 182930](https://github.com/user-attachments/assets/e5f77c46-b7fa-4ef5-bd1d-e0d9e12e6532)

![Screenshot 2024-08-29 182957](https://github.com/user-attachments/assets/241c180a-059b-4b62-a9fa-26f58f19fc98)

![Screenshot 2024-08-29 183018](https://github.com/user-attachments/assets/456096ea-1299-4da5-873b-b5a7d69f3a82)

![Screenshot 2024-08-29 183030](https://github.com/user-attachments/assets/58efd59f-6703-4d19-be2e-246f0da66b1f)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
