import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report


DATA_FILE = "creditcard.csv"
FEATURE_FIRST = 'V1'
FEATURE_LAST = 'V28'
TARGET_NAME = 'Class'


def output(name, **kwargs):
	print(f"\n{name}:")
	for k, v in kwargs.items():
		print(f"{k}: {v}") 


def output_metrics(X_train, X_test, y_train, y_test):
	clf = LogisticRegression(C=100000).fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	confusion = confusion_matrix(y_test, y_predicted)
	accuracy = accuracy_score(y_test, y_predicted)
	precision = precision_score(y_test, y_predicted)
	recall = recall_score(y_test, y_predicted)
	f1 = f1_score(y_test, y_predicted)
	output("confusion_matrix",
		confusion_matrix=confusion,
		accuracy=accuracy,
		precision=precision,
		recall=recall,
		f1_score=f1)


def report(X_train, X_test, y_train, y_test):
	clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8, random_state = 0).fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	report = classification_report(y_test, y_predicted)
	output("classification report", report=report)


def cv_scores(X_data, y_data):
	clf = LogisticRegression(C=100000)
	cv_scores = cross_val_score(clf, X_data, y_data)
	output("Cross val score", cross_val_score=cv_scores)


dataset = pd.read_csv(DATA_FILE)
X_data, y_data = dataset.loc[:,FEATURE_FIRST:FEATURE_LAST], dataset[TARGET_NAME]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

count_metrics(X_train, X_test, y_train, y_test)
report(X_train, X_test, y_train, y_test)
cv_scores(X_data, y_data)