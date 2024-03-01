import numpy as np
from numpy import log, dot, e, shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_features=4, n_classes=2)

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

print(X_tr.shape, X_te.shape)


def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:, i] = (X_tr[:, i] - np.mean(X_tr[:, i])) / np.std(X_tr[:, i])


def f1_score(y, y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


class LogisticRegression:

    def initialize(self, X):
        weights = np.zeros((shape(X)[1] + 1, 1))
        X = np.c_[np.ones((shape(X)[0], 1)), X]
        return weights, X

    def sigmoid(self, z):
        sig = 1 / (1 + e ** (-z))
        return sig

    def fit(self, X, y, alpha=0.001, iter=10000):
        params, X = self.initialize(X)

        def cost(theta): # uses the log-loss: [y(log(y) + (1-y)log(1-y)]
            z = dot(X, theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1 - y).T.dot(log(1 - self.sigmoid(z)))
            cost = -(cost0 + cost1) / len(y)
            return cost

        cost_list = np.zeros(
            iter,
        )
        for i in range(iter): # gradient descent
            params -= alpha * dot(
                X.T, self.sigmoid(dot(X, params)) - np.reshape(y, (len(y), 1))
            )

            cost_list[i] = cost(params)
        self.params = params
        return cost_list

    def predict(self, X):
        z = dot(self.initialize(X)[1], self.params)
        lis = []
        for i in self.sigmoid(z):
            lis.append(1) if i > 0.5 else lis.append(0) # Keeping the threshold 0.5
        return lis


standardize(X_tr)
standardize(X_te)
obj1 = LogisticRegression()
model = obj1.fit(X_tr, y_tr)
y_train = obj1.predict(X_tr)
y_pred = obj1.predict(X_te)

f1_tr = f1_score(y_tr, y_train)
f1_te = f1_score(y_te, y_pred)
print(f1_tr)
print(f1_te)
