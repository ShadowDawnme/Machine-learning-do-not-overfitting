
from sklearn.cross_validationi import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import RFE



def get_data():
    data = ()# remeber to load in the data and change the path
    x = data['data']
    y = data['target']
    return x, y




def build_model(x, y, no_features):
    model = LinearRegression(normalize=True, fit_intercept=True)
    rfe_model = RFE(estimator=model, n_features_to_select=no_features)
    rfe_model.fit(x, y)
    return rfe_model






def model_worth(true_y, predicted_y):


    return mean_squared_error(true_y, predicted_y)



def plot_residual(y, predicted_y):
    plt.cla()
    plt.xlabel('predicted y')
    plt.ylabel('residual')
    plt.title('residual plot')
    plt.figure1(1)
    diff = y - predicted_y
    plt.plot(predicted_y, diff, 'go')
    plt.show()


if __name__ == "__main__":
    x, y = get_data()

    x_train, x_test_all, y_train, y_test_all = train_test_split(x, y, \
                                                                test_size=0.3, random_state=9)
    x_dev, x_test, y_dev, y_test = train_test_split(x_test_all, y_test_all, \
                                                    test_size=0.3, random_state=9)

    poly_features = PolynomialFeatures(interaction_only=True)
    x_train_poly = poly_features.fit_transform(x_train)
    x_dev_poly = poly_features.fit_transform(x_dev)
    choosen_model = build_model(x_train_poly, y_train, 20)
    predicted_y = choosen_model.predict(x_train_poly)
    mse = model_worth(y_train, predicted_y)

    x_test_poly = poly_features.fit_transform(x_test)
    predicted_y = choosen_model.predict(x_test_poly)

    model_worth(y_test, predicted_y)