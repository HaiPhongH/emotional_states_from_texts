from phobert_model import TrainingBert
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.svm import SVC

import pandas as pd
import joblib
import numpy as np

training_bert = TrainingBert()

# load data for training process
print("Loading data...")
train_file = 'data/data1.csv'
list_review, list_label = training_bert.load_data(train_file)

# using PhoBERT to extract festure vectors from training data
print("Creating BERT-features...")
features = training_bert.create_feature_vector(list_review)

# split data training into 2 set: training data and validation data
x_train, x_test, y_train, y_test = train_test_split(features, list_label, test_size = 0.2, random_state = 0)

# using SVM for classification
classifier = SVC(kernel = 'linear', probability = True)
classifier.fit(features, list_label)


# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(
#         SVC(), tuned_parameters, scoring='%s_macro' % score
#     )
#     clf.fit(x_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()

save model
dump(classifier, 'src/pretrained_model.pkl')