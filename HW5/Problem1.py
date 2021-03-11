import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialApproximation:
    def __init__(self, xy, z):
        self.xy = xy
        self.z = z
        self.model = LinearRegression(fit_intercept=False)
        self.coeffs = None
        self.degree = None

    def polynomial_features(self, xy):
        assert (self.degree is not None)
        xy_poly = PolynomialFeatures(self.degree).fit_transform(xy)
        return xy_poly

    def set_degree(self, degree):
        self.degree = degree
        self.model.fit(self.polynomial_features(self.xy), self.z)
        self.coeffs = self.model.coef_
        return self

    def P(self, xy):
        return np.dot(self.polynomial_features(xy), self.coeffs.T)

    def error(self, xy, z):
        z_predictions = self.P(xy)
        return z - z_predictions

    def mse(self, xy, z):
        return np.average(self.error(xy, z)**2)

def main():
    df = pd.read_csv("HW5_data.csv", index_col=[0])
    # Split the dataset into train, validation and test with the ratio of 0.6:0.2:0.2
    train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
    train_data, validation_data = train_test_split(train_data, test_size = 0.25, random_state = 42)

    xy_train = train_data[['X', 'Y']]
    xy_validation = validation_data[['X', 'Y']]
    xy_test = test_data[['X', 'Y']]
    z_train = train_data[['Z']]
    z_validation = validation_data[['Z']]
    z_test = test_data[['Z']]

    poly_approx = PolynomialApproximation(xy_train, z_train)
    mean_squared_errors = []
    degrees = [2, 3, 4, 5, 6, 7]
    for degree in degrees:
        poly_approx.set_degree(degree)
        mse = poly_approx.mse(xy_validation, z_validation)
        mean_squared_errors.append(mse)

    best_D = np.argmin(mean_squared_errors) + 2
    poly_approx.set_degree(best_D)
    print("The Best D:", best_D)
    print("MSE of test data:", poly_approx.mse(xy_test, z_test))
    print("The coefficients for the best D:", poly_approx.coeffs)

if __name__ == "__main__":
    main()