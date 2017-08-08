# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition   import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.svm         import SVC

# This file includes classifier code

# To use:
# 1) set X, y
# 2) feed a pd.DataFrame to construct X_train, X_test, y_train, y_test
# 3) call desired models with parameters

TEST_PERCENT = 0.20
PCA_FEATURES = None

pca = None
X_labels = None
y_labels = "Weather"
X_train = None
X_test  = None
y_train = None
y_test  = None
test_index = None

def feed(df):
    global X_train, X_test, y_train, y_test
    global pca, test_index
    X = df[X_labels]
    y = df[y_labels]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENT)
    test_index = X_test.index
    if PCA_FEATURES is not None:
        pca = PCA(PCA_FEATURES).fit(X_train, y_train)
        X_train = pca.transform(X_train)
        X_test  = pca.transform(X_test)

def knn(n=1, post=False):
    model = KNeighborsClassifier(n_neighbors=n)
    name = "K Nearest Neighbours (n=%d)" %n
    return use_model(model, name, post)

def svm(C, gamma, post=False):
    model = SVC(C=C,gamma=gamma)
    name = "Support Vector Machine (C=%g, gamma=%g)" % (C, gamma)
    return use_model(model, name, post)

def bayes(post=False):
    model = GaussianNB()
    return use_model(model, "Naive Bayes", post)

def use_model(model, name, post=False):
    model.fit(X_train, y_train)
    if post:
        print(name + " scored: %.2f%%" % (model.score(X_test, y_test)))
    return model
