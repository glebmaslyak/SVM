import numpy as np
import random
from cvxopt import matrix, solvers
from sklearn.svm import SVC, LinearSVC
import time
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, C=1, method='primal', kernel=None, gamma=None):
        self.C = C
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.A = None

    def compute_primal_objective(self, X, y):
        if not self.w is None:
            assert(X.shape[0] == y.shape[0])
            M = 0
            if self.w.shape[0] == X.shape[1]:
                for i in range(X.shape[0]):
                    M += max(0, 1 - y[i] * (np.sum(self.w.T * X[i]) + self.w0))
                return 0.5 * np.sum(self.w ** 2) + M
            else:
                assert(X.shape[1] == self.w.shape[0]-1)
                on = np.ones((X.shape[0], 1))
                X_n = np.hstack((on, X))
                for i in range(X.shape[0]):
                    M += max(0, 1 - y[i] * np.sum(self.w.T * X_n[i]))
                return 0.5 * np.sum(self.w[1:] ** 2) + M
        else:
            assert (not self.w is None)

    def compute_dual_objective(self, X, y):
        assert(X.shape[0] == y.shape[0])
        if self.A is None:
            assert (not self.A is None)
        else:
            assert(X.shape[0] == self.A.shape[0])
            K = np.zeros((X.shape[0], X.shape[0]))
            if self.kernel == "rbf":
                for i in range(X.shape[0]):
                    for j in range(X.shape[0]):
                        K[i, j] = np.exp(0.5 / (self.gamma) * np.sum(X[i] - X[j]))
            else:
                K = np.sum(self.X[np.newaxis:] * X[:, np.newaxis], axis=2)
            return np.sum(self.A) - 0.5 * (self.A.T @ K @ self.A)

    def fit(self, X, y, tol=0.001, max_iter=10000, verbose=False, stop_criterion='objective', batch_size=50, lamb=0.5, alpha=1, beta=1):
        assert(X.shape[0] == y.shape[0])
        dictionary = {"time": None, "objective_curve": None, "status": None}
        obj = []
        start = time.clock()
        if self.method == 'stoch_subgradient':
            assert(max_iter > 0)
            if batch_size <= 0 or batch_size > X.shape[0]:
                assert (batch_size > 0)
                assert (batch_size <= X.shape[0])
            lst = []
            sch = 0
            while sch < batch_size:
                i = random.randrange(X.shape[0])
                if not i in lst:
                    lst.append(i)
                    sch += 1
            lst.sort()
            lst = np.array(lst)
            w = np.zeros((X.shape[1] + 1, 1))
            on = np.ones((X.shape[0], 1))
            X_new = np.hstack((on, X))
            self.w = w.copy()
            q = self.compute_primal_objective(X, y)
            if verbose:
                print("итерация ", 0, " целевая переменная", q)
            obj.append(q)
            eps = 0
            for i in range(batch_size):
                eps += max(0, 1 - y[lst[i]] * X_new[lst[i]].dot(self.w))
            func_q = (1 - lamb) * q + lamb * eps
            s = 0
            for it in range(1, max_iter):
                M = np.zeros((X.shape[1] + 1, 1))
                for i in range(batch_size):
                    if 1 - y[lst[i]] * np.sum(w.T * X_new[lst[i]]) >= 0:
                        M -= self.C * (alpha / it ** beta) * y[lst[i]] * X_new[lst[i]].reshape(X_new.shape[1], 1)
                self.w = w - (alpha / it ** beta) * w - M
                if stop_criterion == 'argument':
                    obj.append(self.compute_primal_objective(X, y))
                    if np.sum(w ** 2) ** 0.5 < tol:
                        dictionary["status"] = 0
                        dictionary["objective_curve"] = obj
                        dictionary["time"] = time.clock() - start
                        break
                elif stop_criterion == 'objective':
                    q = self.compute_primal_objective(X, y)
                    obj.append(q)
                    eps = 0
                    for i in range(batch_size):
                        eps += max(0, 1 - y[lst[i]] * X_new[lst[i]].dot(self.w))
                    func_q1 = (1 - lamb) * q + lamb * eps
                    if abs(func_q1 - func_q) < tol:
                        dictionary["status"] = 0
                        dictionary["objective_curve"] = obj
                        dictionary["time"] = time.clock() - start
                        break
                    else:
                        func_q = func_q1
                else:
                    assert (stop_criterion in ["objective", "argument"])
                if verbose:
                    print("итерация ", it, " целевая переменная", obj[it])
                w = self.w
                sch = 0
                lst = []
                while sch < batch_size:
                    i = random.randrange(X.shape[0])
                    if not i in lst:
                        lst.append(i)
                        sch += 1
                lst = np.array(lst)
            self.w0 = self.w[0, 0]
            self.w = self.w[1:]
            dictionary["status"] = 1 if it == max_iter else 0
            dictionary["objective_curve"] = obj
            dictionary["time"] = time.clock() - start
        elif self.method == 'subgradient':
            w = np.zeros((X.shape[1] + 1, 1))
            self.w = w.copy()
            x0 = np.ones((X.shape[0], 1))
            X_new = np.hstack((x0, X))
            s = 0
            t = self.compute_primal_objective(X, y)
            if verbose:
                print("итерация ", 0, " целевая переменная ", t)
            obj.append(t)
            while True:
                if s >= max_iter-1:
                    self.w0 = self.w[0, 0]
                    self.w = self.w[1:]
                    dictionary["status"] = 1
                    dictionary["objective_curve"] = obj
                    dictionary["time"] = time.clock() - start
                    break
                s += 1
                M = np.zeros((X.shape[1] + 1, 1))
                for i in range(X.shape[0]):
                    if 1 - y[i] * np.sum(w.T * X_new[i]) >= 0:
                        M -= self.C * (alpha / s ** beta) * y[i] * X_new[i].reshape(X_new.shape[1], 1)
                self.w = w - (alpha / s ** beta) * w - M
                t1 = self.compute_primal_objective(X, y)
                obj.append(t1)
                if verbose:
                    print("итерация ", s, "целевая функция", t1)
                if stop_criterion == 'objective':
                    if abs(t1 - t) < tol:
                        self.w0 = self.w[0, 0]
                        self.w = self.w[1:]
                        dictionary["status"] = 0
                        dictionary["objective_curve"] = obj
                        dictionary["time"] = time.clock() - start
                        break
                    else:
                        w = self.w
                        t = t1
                elif stop_criterion == 'argument':
                    if np.sum((self.w - w) ** 2) ** 0.5 < tol:
                        dictionary["status"] = 0
                        dictionary["objective_curve"] = obj
                        self.w0 = self.w[0, 0]
                        self.w = self.w[1:]
                        dictionary["time"] = time.clock() - start
                        break
                    else:
                        w = self.w
                else:
                    assert (stop_criterion in ["objective", "argument"])
        elif self.method == 'primal':
            on = np.ones((X.shape[0], 1))
            X_new = np.hstack((on, X))
            q = np.zeros((X_new.shape[1] + X.shape[0], 1))
            for i in range(X_new.shape[0]):
                q[i + X_new.shape[1], 0] = self.C
            P = np.zeros((X_new.shape[1] + X.shape[0], X_new.shape[1] + X.shape[0]))
            for i in range(1, X_new.shape[1]):
                P[i, i] = 1
            G = X_new * y.reshape(y.shape[0], 1).copy()
            G = np.hstack((G, np.identity(X.shape[0])))
            G = G * (-1)
            G1 = np.hstack((np.zeros((X.shape[0], X_new.shape[1])), (-1) * np.identity(X.shape[0])))
            h = np.ones((X.shape[0], 1)) * (-1)
            G = np.vstack((G, G1))
            h1 = np.zeros((X.shape[0], 1))
            h = np.vstack((h, h1))
            q = matrix(q, tc='d')
            P = matrix(P, tc='d')
            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            sol = solvers.qp(P, q, G, h)
            a = sol['x']
            self.w = np.array(a[1:X_new.shape[1]])
            self.w0 = a[0, 0]
            dictionary["time"] = time.clock() - start
            dictionary["status"] = 0 if sol['status'] == 'optimal' else 1
        elif self.method == 'dual':
            self.X = X
            self.y = y
            if not self.kernel is None:
                samples, features = X.shape
                K = np.zeros((samples, samples))
                if self.kernel == 'linear':
                    for i in range(samples):
                        for j in range(samples):
                            K[i, j] = np.sum(X[i] * X[j])
                elif self.kernel == 'rbf':
                    if self.gamma is None:
                        assert (not self.gamma is None)
                    if self.gamma <= 0:
                        assert (self.gamma > 0)
                    K = np.exp((-0.5 / self.gamma) * np.sum((X[np.newaxis:] - X[:, np.newaxis]) ** 2, axis=2))
                P = matrix(np.outer(y, y) * K, tc='d')
                q = matrix(np.ones(samples) * -1, tc='d')
                A = matrix(y, (1, samples), tc='d')
                b = matrix(0.0, tc='d')
                tmp1 = np.diag(np.ones(samples) * -1)
                tmp2 = np.identity(samples)
                G = matrix(np.vstack((tmp1, tmp2)), tc='d')
                tmp1 = np.zeros(samples)
                tmp2 = np.ones(samples) * self.C
                h = matrix(np.hstack((tmp1, tmp2)), tc='d')
                solution = solvers.qp(P, q, G, h, A, b)
                dictionary["time"] = time.clock() - start
                dictionary["status"] = 0 if solution['status'] == 'optimal' else 1
                self.A = np.array(solution['x'])
            else:
                assert (not self.kernel is None)
        elif self.method == "liblinear":
            self.clf = LinearSVC(C=self.C, tol=tol, verbose=verbose, max_iter=max_iter)
            self.clf.fit(X, y[:,0])
            self.w = self.clf.coef_.T
            self.w0 = self.clf.intercept_
            dictionary["time"] = time.clock() - start
        elif self.method == "libsvm":
            self.clf = SVC(C=self.C, kernel=self.kernel, gamma=float(self.gamma), tol=tol, verbose=bool(verbose), max_iter=max_iter)
            self.clf.fit(X, y[:,0])
            self.A = np.zeros((X.shape[0], 1))
            self.A[self.clf.support_] = self.clf.dual_coef_.reshape(self.clf.support_.shape[0], 1)
            dictionary["time"] = time.clock() - start
        else:
            assert ("Please, choose another method")
        return dictionary

    def predict(self, X_test, return_classes=False):
        if return_classes:
            if self.method == "dual":
                assert(not self.A is None)
                assert(X_test.shape[1] == self.X.shape[1])
                if self.kernel == "linear":
                    K = np.sum((X_test[:, np.newaxis] * self.X[np.newaxis:]), axis=2)
                    w = np.sum(self.A*self.y*self.X, axis=0)
                    ind = self.A.argmax()
                    w0 = np.sum(w * self.X[ind]) - self.y[ind]
                    pred = np.sum(w*X_test, axis=1) - w0
                elif self.kernel == "rbf":
                    K = np.exp((-0.5 / self.gamma) * np.sum((X_test[:, np.newaxis] - self.X[np.newaxis:]) ** 2, axis=2))
                    pred = np.sum(self.y.reshape(1, self.X.shape[0]) * (self.A.reshape(1, self.X.shape[0]) * K), axis=1)
                return np.sign(pred)
            if self.method == "subgradient" or self.method == "stoch_subgradient" or self.method == "primal":
                assert(not self.w is None)
                assert(self.w.shape[0] == X_test.shape[1])
                pred = np.sum(self.w.T * X_test, axis=1) + self.w0
                return np.sign(pred)
            if self.method == "liblinear" or self.method == "libsvm":
                return self.clf.predict(X_test)
        else:
            if method == "dual":
                self.w = compute_w(self.X, self.y)
            assert(not self.w is None)
            assert(self.w.shape[0] == X_test.shpe[1])
            assert(method in ["stoch_subgradient","subgradient", "dual","primal", "liblinear"])
            return np.sum(self.w.T*X_test, axis=1) + self.w0


    def compute_support_vectors(self, X):
        if self.method == "libsvm":
            return self.clf.support_vectors_
        if not self.A is None:
            if X.shape[0] != self.A.shape[0]:
                assert (X.shape[0] == self.A.shape[0])
            else:
                a = np.zeros(self.A.shape[0], dtype=float)
                for i in range(a.shape[0]):
                    a[i] = self.A[i, 0]
                b = a.max()
                return X[a > b/100]
        else:
            assert (not self.A is None)

    def compute_w(self, X, y):
        if not self.A is None:
            if self.kernel == "linear":
                w = np.sum(X * self.A * y.reshape(X.shape[0], 1), axis=0)
                w0 = 0
                for i in range(X.shape[0]):
                    if self.A[i, 0] != 0:
                        w0 = y[i] - np.sum(w * X[i])
                        if y[i] * (np.sum(w * X[i]) + w0) == 1.0:
                            break
                return np.hstack((w0, w))
            else:
                assert(self.kernel == "linear")
        else:
            assert (not self.A is None)

def visualize(X, y, alg_svm, show_vectors=False):
    n_classes = 2
    plot_colors = "bry"
    plot_step = 0.05
    assert(X.shape[0] != 2)
    if not show_vectors:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
        Z = alg_svm.predict(np.c_[xx.ravel(), yy.ravel()], return_classes=True)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.axis("tight")
        for i, color in zip([-1,1], plot_colors):
            idx = np.where(y.reshape(1, y.shape[0])[0] == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
        plt.axis("tight")
    else:
        if alg_svm.A is None:
            assert(not alg_svm is None)
        else:
            X_new = alg_svm.compute_support_vectors(X)
            y3 = np.array([[1]])
            if len(X_new.shape) == 1:
                        y3 = y[(X == X_new).all(axis=1)]
            else:
                for i in X_new:
                    y3 = np.vstack((y3, y[(X == i).all(axis=1)]))
            y_new = y3[1:]
            x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
            y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
            Z = alg_svm.predict(np.c_[xx.ravel(), yy.ravel()], return_classes=True)
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.axis("tight")
            for i, color in zip([-1,1], plot_colors):
                idx = np.where(y_new.reshape(1, y_new.shape[0])[0] == i)
                plt.scatter(X_new[idx, 0], X_new[idx, 1], c=color, cmap=plt.cm.Paired)
            plt.axis("tight")
