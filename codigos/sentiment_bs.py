import numpy as np
import pandas as pd
import math
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances

def func_gb(vf, n, r, rho, delta=0.05):
    return vf + math.sqrt(
        (4.0 / n) *
        (((r**2 * math.log2(n)**2) / rho**2) + math.log2(1 / delta)))


data_negative = np.genfromtxt(
    './sentiment/CLEAN_buscape2.nc', delimiter=';')
data_positve = np.genfromtxt(
    './sentiment/CLEAN_buscape2.pc', delimiter=';')

neg = np.zeros((data_negative.shape[0], 1), dtype=np.int)
pos = np.ones((data_positve.shape[0], 1), dtype=np.int)

data_negative = np.concatenate((data_negative, neg), 1)
data_positve = np.concatenate((data_positve, pos), 1)

data = np.concatenate((data_negative, data_positve), 0)

target = data[:, data.shape[1] - 1]
data = np.delete(data, data.shape[1] - 1, 1)

data = Imputer().fit_transform(data)

state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(state)
np.random.shuffle(target)

print("Original shape: " + str(data.shape))

# print(set(target))
print(target)

x_data, y_data, x_lbl, y_lbl = train_test_split(
    data, target, test_size=0.5, random_state=1)

scaler = StandardScaler()

# Transform train data between 0 and 1
scaler.fit(x_data)
x_data = scaler.transform(x_data)
y_data = scaler.transform(y_data)

# Applying PCA with .95% of variance
pca = PCA(.95)
pca.fit(x_data)
x_data = pca.transform(x_data)
y_data = pca.transform(y_data)

print("PCA shape: " + str(x_data.shape))


param_grid = [
    {
        'coef0': [0, 1],
        'gamma': [0.01, 0.001, 0.0001],
        'kernel': ['linear']
    },
    {
        'coef0': [0, 1],
        'degree': [2, 3, 4, 5],
        'gamma': [0.01, 0.001, 0.0001],
        'kernel': ['poly']
    },
    {
        'coef0': [0, 1],
        'gamma': [0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    },
]

cv = KFold(n_splits=10)

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), param_grid, cv=10, scoring='%s' % score, verbose=1)

    clf.fit(x_data, x_lbl)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_lbl, clf.predict(y_data)
    print(classification_report(y_true, y_pred))
    print()

    clf2 = SVC(coef0=1, degree=2, gamma=0.001, kernel='poly')
    clf2.fit(x_data, x_lbl)
    gb = func_gb(1 - clf.score(y_data, y_lbl), x_data.shape[0],
                 np.max(euclidean_distances(x_data)), -1 * clf2.intercept_,
                 0.05)
    print("ACcuracy: " + str(gb))

