import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

DATA_FILE = "creditcard.csv"
FEATURE_FIRST = 'V1'
FEATURE_LAST = 'V28'
TARGET_NAME = 'Class'


def output(name, **kwargs):
	print(f"\n{name}:")
	for k, v in kwargs.items():
		print(f"{k}: {v}") 


def logistic_regression(X_train, X_test, y_train, y_test):
	clf = LogisticRegression(C=100000).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("logistic_regression", coef=clf.coef_, intercept=clf.intercept_, training_score=training_score, test_score=test_score)



def linearSVC(X_train, X_test, y_train, y_test):
	clf = LinearSVC(C=100000).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("linearSVC", coef=clf.coef_, intercept=clf.intercept_, training_score=training_score, test_score=test_score)


dataset = pd.read_csv(DATA_FILE)
X_data, y_data = dataset.loc[:,FEATURE_FIRST:FEATURE_LAST], dataset[TARGET_NAME]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

logistic_regression(X_train, X_test, y_train, y_test)
linearSVC(X_train, X_test, y_train, y_test)