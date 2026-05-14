# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as pandas, numpy, matplotlib, and sklearn modules.

2.Read the dataset from the CSV file using pandas and display the contents using head() and tail() functions.

3.Separate the dataset into independent variable (X) and dependent variable (y).

4.Split the dataset into training and testing sets using the train_test_split() function.

5.Create and train the Linear Regression model using the training data, then predict the output for the test data.

6.Evaluate the model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), and visualize the results using graphs.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jeevan Vishal.G.D
RegisterNumber:212224240062
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('student_scores.csv')

print("Head Values:\n", df.head())
print("\nTail Values:\n", df.tail())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("\nX Values:\n", X)
print("\ny Values:\n", y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values:\n", y_pred)
print("\nActual Values:\n", y_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title("Training Set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test), color='blue')
plt.title("Testing Set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)
```

## Output:
HEAD VALUE
<img width="217" height="158" alt="Screenshot 2026-04-27 154300" src="https://github.com/user-attachments/assets/04dd1e5a-be58-4758-9f64-311b6202340f" />
TAIL VALUE
<img width="217" height="155" alt="Screenshot 2026-04-27 154306" src="https://github.com/user-attachments/assets/e5d3aea0-ee87-499b-a003-50e677ebc48e" />
X VALUE
<img width="184" height="562" alt="Screenshot 2026-04-27 154540" src="https://github.com/user-attachments/assets/98aee6c7-b724-4ac1-a41f-c457889427f1" />
Y VALUE
<img width="730" height="68" alt="Screenshot 2026-04-27 154549" src="https://github.com/user-attachments/assets/a53aa9d0-85a0-4c04-9412-c40511bb9c83" />
PREDICTED VALUES
<img width="776" height="100" alt="Screenshot 2026-04-27 154701" src="https://github.com/user-attachments/assets/8741f3bd-e87f-4cb0-a32a-ee67e0d5e5f7" />
ACTUAL VALUES
<img width="476" height="48" alt="Screenshot 2026-04-27 154716" src="https://github.com/user-attachments/assets/d90af513-820e-498c-97af-69e221fb441b" />
TRAINING GRAPH
<img width="855" height="560" alt="Screenshot 2026-04-27 154828" src="https://github.com/user-attachments/assets/f469d7c6-3601-4c98-ba66-723e24f0e50d" />
TESTING GRAPH
<img width="753" height="601" alt="Screenshot 2026-04-27 154837" src="https://github.com/user-attachments/assets/bbd31a06-9ebd-4c11-8e7c-8e70b2c1ff5c" />
ERROR METRICS
<img width="309" height="90" alt="Screenshot 2026-04-27 154844" src="https://github.com/user-attachments/assets/53ca3197-7220-4869-ad76-1f2cb98929aa" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
