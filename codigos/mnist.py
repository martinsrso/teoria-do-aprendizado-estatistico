from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def func_gb(vf, n, r, rho, delta=0.05):
    return vf + math.sqrt(
        (4.0 / n) *
        (((r**2 * math.log2(n)**2) / rho**2) + math.log2(1 / delta)))


N = 12665

#from sklearn import datasets

#d = datasets.load_digits()
#n = len(d.images)
#mnistdata = d.images.reshape((n, -1))
#mnisttarget = d.target
mnist = fetch_mldata('MNIST original')

mnist.data = mnist.data[:N]
mnist.target = mnist.target[:N]

state = np.random.get_state()
np.random.shuffle(mnist.data)
np.random.set_state(state)
np.random.shuffle(mnist.target)

print("Original shape: " + str(mnist.data.shape))

print(set(mnist.target))

x_data, y_data, x_lbl, y_lbl = train_test_split(
    mnist.data, mnist.target, test_size=0.5, random_state=1)

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

print("PCA shape: " +
      str(x_data.shape))

param_grid = [
    {'coef0': [0, 1], 'gamma': [
        0.01, 0.001, 0.0001], 'kernel': ['linear']},
    {'coef0': [0, 1], 'degree': [2, 3, 4, 5],
        'gamma': [0.01, 0.001, 0.0001], 'kernel': ['poly']},
    {'coef0': [0, 1], 'gamma': [
        0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]

cv = KFold(n_splits=10)

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(),
                       param_grid,
                       cv=10,
                       scoring='%s' % score,
                       verbose=1)

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
    clf2 = SVC(coef0=1, degree=3, gamma=0.001, kernel='poly')
    clf2.fit(x_data, x_lbl)
    gb = func_gb(1 - clf.score(y_data, y_lbl), x_data.shape[0],
                 np.max(euclidean_distances(x_data)), -1 * clf2.intercept_,
                 0.05)
    print("ACcuracy: " + str(gb))
