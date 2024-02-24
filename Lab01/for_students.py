import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

m = x_train.size
x_matrix = np.concatenate((np.ones((m, 1)), x_train.reshape(-1,1)), 1)
y_matrix = y_train.reshape(-1, 1)
theta_matrix = np.linalg.inv(x_matrix.T.dot(x_matrix)).dot(x_matrix.T).dot(y_matrix)
theta_best = theta_matrix.flatten()

print(theta_best)

# TODO: calculate error (using only the train data)

mse = (1 / m) * sum((theta_best[1]*x + theta_best[0] - y) ** 2 for x,y in zip(x_train, y_train))
print("MSE", mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_mean = np.mean(x_train)
x_std  = np.std(x_train)
y_mean = np.mean(y_train)
y_std  = np.std(y_train)
x_train = np.fromiter(map(lambda x: (x - x_mean) / x_std, x_train), dtype=float)
y_train = np.fromiter(map(lambda y: (y - y_mean) / y_std, y_train), dtype=float)
x_test = np.fromiter(map(lambda x: (x - x_mean) / x_std, x_test), dtype=float)
y_test = np.fromiter(map(lambda y: (y - y_mean) / y_std, y_test), dtype=float)

# recalculate
x_matrix = np.concatenate((np.ones((m, 1)), x_train.reshape(-1,1)), 1)
y_matrix = y_train.reshape(-1, 1)

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(1, 2).reshape(-1, 1)
learning_rate = 0.001

for i in range(0, 100000):
    gradientMse = (2/m) * x_matrix.T.dot(x_matrix.dot(theta_best) - y_matrix)
    theta_best = theta_best - learning_rate * gradientMse

theta_best = theta_best.flatten()
print(theta_best)

# TODO: calculate error

mse = (1 / m) * sum((theta_best[1]*x + theta_best[0] - y) ** 2 for x,y in zip(x_train, y_train))
print("MSE", mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()