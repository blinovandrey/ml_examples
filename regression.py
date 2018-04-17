import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures


DATA_FILE = "fish_length.txt"
FEATURES_NAMES = ['age', 'temp']
TARGET_NAME = 'length'


def output(name, **kwargs):
	print(f"\n{name}:")
	for k, v in kwargs.items():
		print(f"{k}: {v}") 


def knn(X_train, X_test, y_train, y_test):
	knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
	training_score = knn.score(X_train, y_train)
	test_score = knn.score(X_test, y_test)
	output("KNN", training_score=training_score, test_score=test_score)


def linear_regression(X_train, X_test, y_train, y_test):
	linreg = LinearRegression().fit(X_train, y_train)
	training_score = linreg.score(X_train, y_train)
	test_score = linreg.score(X_test, y_test)
	output("linear_regression", coef=linreg.coef_, intercept=linreg.intercept_, training_score=training_score, test_score=test_score)


def ridge_regression(X_train, X_test, y_train, y_test):
	linridge = Ridge(alpha=20.0).fit(X_train, y_train)
	training_score = linridge.score(X_train, y_train)
	test_score = linridge.score(X_test, y_test)
	output("ridge_regression", coef=linridge.coef_, intercept=linridge.intercept_, training_score=training_score, test_score=test_score)


def ridge_regression_scaled(X_train, X_test, y_train, y_test):
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)
	linridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
	training_score = linridge.score(X_train_scaled, y_train)
	test_score = linridge.score(X_test_scaled, y_test)
	output("ridge_regression_scaled", coef=linridge.coef_, intercept=linridge.intercept_, training_score=training_score, test_score=test_score)


def lasso_regression_scaled(X_train, X_test, y_train, y_test):
	scaler = MinMaxScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)
	linridge = Lasso(alpha=1.0).fit(X_train_scaled, y_train)
	training_score = linridge.score(X_train_scaled, y_train)
	test_score = linridge.score(X_test_scaled, y_test)
	output("lasso_regression_scaled", coef=linridge.coef_, intercept=linridge.intercept_, training_score=training_score, test_score=test_score)


def linear_regression_polynomial(X_train, X_test, y_train, y_test):
	poly = PolynomialFeatures(degree=2)
	X_train_poly = poly.fit_transform(X_train)
	X_test_poly = poly.fit_transform(X_test)
	linreg = LinearRegression().fit(X_train_poly, y_train)
	training_score = linreg.score(X_train_poly, y_train)
	test_score = linreg.score(X_test_poly, y_test)
	output("linear_regression_polynomial", coef=linreg.coef_, intercept=linreg.intercept_, training_score=training_score, test_score=test_score)


dataset = pd.read_table(DATA_FILE, delim_whitespace=True, comment='#')
X_data, y_data = dataset[FEATURES_NAMES], dataset[TARGET_NAME]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

knn(X_train, X_test, y_train, y_test)
linear_regression(X_train, X_test, y_train, y_test)
ridge_regression(X_train, X_test, y_train, y_test)
ridge_regression_scaled(X_train, X_test, y_train, y_test)
lasso_regression_scaled(X_train, X_test, y_train, y_test)
linear_regression_polynomial(X_train, X_test, y_train, y_test)