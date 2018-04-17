import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


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


def linear_SVC(X_train, X_test, y_train, y_test):
	clf = LinearSVC(C=100000).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("linearSVC", coef=clf.coef_, intercept=clf.intercept_, training_score=training_score, test_score=test_score)


def rbf_SVC(X_train, X_test, y_train, y_test):
	clf = SVC(C=100000).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("linearSVC", training_score=training_score, test_score=test_score)


def decision_tree(X_train, X_test, y_train, y_test):
	clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=0).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("Decision tree", feature_importances=clf.feature_importances_, training_score=training_score, test_score=test_score)


def gaussian_nb(X_train, X_test, y_train, y_test):
	clf = GaussianNB().fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("Gaussian NB", training_score=training_score, test_score=test_score)


def random_forest(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(max_features=8, random_state=0).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("Random forest", training_score=training_score, test_score=test_score)


def gradient_boosting(X_train, X_test, y_train, y_test):
	clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=2, random_state=0).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("Gradient Boosting", training_score=training_score, test_score=test_score)


def mlp_neural_network(X_train, X_test, y_train, y_test):
	clf = MLPClassifier(
		solver='lbfgs',
		activation='tanh',
    	alpha=1,
        hidden_layer_sizes=[100, 100],
        random_state=0).fit(X_train, y_train)
	training_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	output("MLP neural network", training_score=training_score, test_score=test_score)


dataset = pd.read_csv(DATA_FILE)
X_data, y_data = dataset.loc[:,FEATURE_FIRST:FEATURE_LAST], dataset[TARGET_NAME]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

logistic_regression(X_train, X_test, y_train, y_test)
linear_SVC(X_train, X_test, y_train, y_test)
rbf_SVC(X_train, X_test, y_train, y_test)
decision_tree(X_train, X_test, y_train, y_test)
gaussian_nb(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)
gradient_boosting(X_train, X_test, y_train, y_test)
mlp_neural_network(X_train, X_test, y_train, y_test)
