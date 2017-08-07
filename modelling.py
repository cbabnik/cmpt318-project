# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.svm         import SVC

# This file includes classifier code

# To use:
# 1) set X, y
# 2) feed a pd.DataFrame to construct X_train, X_test, y_train, y_test

TEST_PERCENT = 0.20

X_labels = []
y_labels = "Weather"
X_train = []
X_test  = []
y_train = []
y_test  = []

def feed(df):
    global X_train, X_test, y_train, y_test
    X = df[X_labels].values
    y = df[y_labels].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENT)

def knn(n=1, post=False, write=False):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    if post:
        print("K Nearest Neighbors (n=%d) scored: %f \% accuracy"
              % (n, model.score(X_test, y_test)))
    if write:
        pass
    return model

def svm(C, gamma=None, post=False, write=False):
    if gamma==None:
        model = SVC(C=C)
    else:
        model = SVC(C=C,gamma=gamma)
    model.fit(X_train, y_train)
    if post:
        print("Support Vector Machine (C=%g, gamma=%g) scored: %f \% accuracy"
              % (C, gamma, model.score(X_test, y_test)))
    if write:
        pass
    return model

def bayes(post=False, write=False):
    model = GaussianNB()
    model.fit(X_train, y_train)
    if post:
        print("Naive Bayes scored: %f \% accuracy" % (model.score(X_test, y_test)))
    if write:
        pass
    return model
