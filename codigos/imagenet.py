import argparse
import os

import math
import cv2
import imutils
import numpy as np
from imutils import paths
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
# import the necessary packages
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import euclidean_distances

def func_gb(vf, n, r, rho, delta=0.05):
    return vf + math.sqrt(
        (4.0 / n) *
        (((r**2 * math.log2(n)**2) / rho**2) + math.log2(1 / delta)))


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("/home/martinsrso/Documents/imgs"))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split("_")[0]

    # extract a color histogram from the image, then update the
    # data matrix and labels list
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

print("Original shape: " + str(np.array(data).shape))

x_data, y_data, x_lbl, y_lbl = train_test_split(
    np.array(data), labels, test_size=0.15, random_state=1)

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

    clf = GridSearchCV(SVC(), param_grid, cv=10, scoring='%s' % score)

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

    clf2 = SVC(coef0=1, degree=5, gamma=0.001, kernel='poly')
    clf2.fit(x_data, x_lbl)
    gb = func_gb(1 - clf.score(y_data, y_lbl), x_data.shape[0],
                 np.max(euclidean_distances(x_data)), -1 * clf2.intercept_,
                 0.05)
    print("ACcuracy: " + str(gb))

