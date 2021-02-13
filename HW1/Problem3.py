import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def predict_target(x, weights):
    return weights @ x.T


def gradient_wrt_weights(x, y, predicted):
    return (-2 / len(y)) * (y - predicted).T @ x


def gradient_descent(x, y):
    # Initialize weights randomly for gradient descent
    x_with_ones = np.insert(x, 0, np.ones(x.shape[0]), 1)
    weights = np.random.rand(x.shape[1] + 1)
    lrate = 0.05  # Learning rate
    for i in range(10000):
        predicted = predict_target(x_with_ones, weights)
        gradient = gradient_wrt_weights(x_with_ones, y, predicted)
        weights = weights - lrate * gradient
    return weights[1:], weights[0]


def linear_regression(x, y):
    # Define the model
    lr = LinearRegression(fit_intercept=True)
    # Fit the model
    lr.fit(x, y)
    return lr.coef_, lr.intercept_


def main():
    # ------------------------------------------- Part 1 ------------------------------------------- #

    df = pd.read_csv('qsar_fish_toxicity.csv')
    # Data cleaning
    tempdf = df
    tempdf[['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NDsCH', 'NDssC', 'MLOGP', 'LC50']] = tempdf[
        '3.26;0.829;1.676;0;1;1.453;3.770'].str.split(';', expand=True)
    tempdf.pop('3.26;0.829;1.676;0;1;1.453;3.770')

    # Separate the independent and dependent variables
    y = tempdf['LC50']
    x = tempdf[['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']]

    coefficients, intercept = linear_regression(x, y)
    print("Part 1")
    print("Linear Regression Coefficients\n")
    print("alphas: ", coefficients)
    print("intercept: ", intercept, "\n")

    # ------------------------------------------- Part 2 ------------------------------------------- #

    x = x.to_numpy(dtype=np.float)
    y = y.to_numpy(dtype=np.float)

    x_with_ones = np.insert(x, 0, np.ones(x.shape[0]), 1)
    print(x_with_ones)
    print(x_with_ones.shape)
    coefficients = np.linalg.inv(x_with_ones.transpose() @ x_with_ones) @ (x_with_ones.transpose() @ y)
    print("Part 2 Solution\n")
    print("alphas: ", coefficients[1:])
    print("intercept: ", coefficients[0])
    print()

    # ------------------------------------------- Part 3 ------------------------------------------- #

    print("Part 3")
    weights, intercept = gradient_descent(x, y)
    print("Gradient Descent Solution\n")
    print("alphas: ", weights)
    print("intercept: ", intercept)


if __name__ == "__main__":
    main()
