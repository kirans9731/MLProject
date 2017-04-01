import numpy

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def make_classifier(classifier_name):

	classifier_pipeline = []

	classifier_pipeline.append(('vectorizer', TfidfVectorizer(analyzer="word", ngram_range=(1,3), stop_words=None, use_idf = True)))
	classifier_pipeline.append(('transformer', TfidfTransformer()))

	if classifier_name == 'LinearSVC':
		classifier_pipeline.append(('clsf', LinearSVC()))
	elif classifier_name == 'MultinomialNB':
		classifier_pipeline.append(('clsf', MultinomialNB()))
	elif classifier_name == 'LogisticRegression':
		classifier_pipeline.append(('clsf', LogisticRegression()))

	return Pipeline(classifier_pipeline)

def kfold_cross_validate(reviews_df, classifiers, k = 10):

	if len(classifiers) == 0:
		classifiers = ['LogisticRegression', 'MultinomialNB', 'LinearSVC']

	data = reviews_df['Text'].tolist()
	clazz = reviews_df['Class'].tolist()

	kfold_res = {}

	for classifier_name in classifiers:

		kf = KFold(n_splits = k)

		precision = 0.0
		recall = 0.0
		f1score = 0.0
		accuracy = 0.0

		for train_id, test_id in kf.split(data):

			train_data = [data[i] for i in train_id]
			train_class = [clazz[i] for i in train_id]

			test_data = [data[i] for i in test_id]
			test_class = [clazz[i] for i in test_id]

			classifier = make_classifier(classifier_name)

			classifier.fit(train_data, train_class)

			prediction = classifier.predict(test_data)

			precision += precision_score(test_class, prediction, average="macro")
			recall += recall_score(test_class, prediction, average="macro")
			f1score += f1_score(test_class, prediction, average="macro")
			accuracy += accuracy_score(test_class, prediction)

		kfold_res[classifier_name] = {'precision': (precision/k), 'recall': (recall/k), 'f1score:':(f1score/k), 'accuracy':(accuracy/k)}

	return kfold_res


