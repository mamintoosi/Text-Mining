#!python
# -*- coding: utf-8 -*-#
"""
Perceptron Algorithm.

@author: Bhishan Poudel

@date:  Oct 31, 2017

https://stackoverflow.com/questions/47213847/how-to-implement-perceptron-in-python

Modified by: M.Amintoosi
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os
import shutil
import random
# np.random.seed(100)


def read_data(infile):
    data = np.loadtxt(infile)
    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def plot_boundary(X, Y, w, epoch):
    try:
        plt.style.use('seaborn-darkgrid')
        # plt.style.use('ggplot')
        # plt.style.available
    except:
        pass

    # Get data for two classes
    idxN = np.where(np.array(Y) == 0)
    idxP = np.where(np.array(Y) == 1)
    XN = X[idxN]
    XP = X[idxP]

    # plot two classes
    plt.scatter(XN[:, 0], XN[:, 1], c='b', marker='_', label="Negative class")
    plt.scatter(XP[:, 0], XP[:, 1], c='r', marker='+', label="Positive class")
    # plt.plot(XN[:,0],XN[:,1],'b_', markersize=8, label="Negative class")
    # plt.plot(XP[:,0],XP[:,1],'r+', markersize=8, label="Positive class")
    plt.title("Perceptron Algorithm iteration: {}".format(epoch))

    # plot decision boundary orthogonal to w
    # w is w2,w1, w0  last term is bias.
    if len(w) == 3:
        a = -w[1] / w[2]
        b = -w[0] / w[2]
        # xx = [-b/a, 0]
        # yy = [0, b]
        # plt.plot(xx,yy,'--g',label='Decision Boundary')
        x_min = min(X[:, 0])
        x_max = max(X[:, 0])
        xx = [x_min, x_max]
        yy = [a*x_min+b, a*x_max+b]
        plt.plot(xx, yy, ':g', label='Decision Boundary')

    if len(w) == 2:
        x2 = [w[0],  w[1],  -w[1],  w[0]]
        x3 = [w[0],  w[1],   w[1], -w[0]]

        x2x3 = np.array([x2, x3])
        XX, YY, U, V = list(zip(*x2x3))
        ax = plt.gca()
        ax.quiver(XX, YY, U, V, scale=1, color='g')

    # Add labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # lines from origin
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.2)
    plt.grid(True)
    plt.legend(loc=1)
    plt.show()

    # Always clost the plot
    plt.close()


def predict(X, w):
    return np.sign(np.dot(X, w))


def plot_contour(X, Y, w, mesh_stepsize):
    try:
        plt.style.use('seaborn-darkgrid')
        # plt.style.use('ggplot')
        # plt.style.available
    except:
        pass
    # Get data for two classes
    idxN = np.where(np.array(Y) == 0)
    idxP = np.where(np.array(Y) == 1)
    XN = X[idxN]
    XP = X[idxP]

    # plot two classes with + and - sign
    fig, ax = plt.subplots()
    ax.set_title('Perceptron Algorithm')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(XN[:, 0], XN[:, 1], 'b_', markersize=8, label="Negative class")
    plt.plot(XP[:, 0], XP[:, 1], 'y+', markersize=8, label="Positive class")
    plt.legend()

    # create a mesh for contour plot
    # We first make a meshgrid (rectangle full of pts) from xmin to xmax and ymin to ymax.
    # We then predict the label for each grid point and color it.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Get 2D array for grid axes xx and yy  (shape = 700, 1000)
    # xx has 700 rows.
    # xx[0] has 1000 values.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

    # Get 1d array for x and y axes
    xxr = xx.ravel()  # shape (700000,)
    yyr = yy.ravel()  # shape (700000,)

    # ones vector
    # ones = np.ones(xxr.shape[0]) # shape (700000,)
    ones = np.ones(len(xxr))  # shape (700000,)

    # Predict the score
    Xvals = np.c_[ones, xxr, yyr]
    scores = predict(Xvals, w)

    # Plot contour plot
    scores = scores.reshape(xx.shape)
    ax.contourf(xx, yy, scores, cmap=plt.cm.Paired)
    # print("xx.shape = {}".format(xx.shape))               # (700, 1000)
    # print("scores.shape = {}".format(scores.shape))       # (700, 1000)
    # print("scores[0].shape = {}".format(scores[0].shape)) # (1000,)

    # show the plot
    plt.savefig("Perceptron.png")
    plt.show()
    plt.close()


def perceptron_sgd(X, Y, epochs, eta):
    """
    X: data matrix without bias.
    Y: target
    """
    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # add bias to X's first column
    ones = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    X1 = np.append(ones, X, axis=1)
    print(X1)
    # w = np.random.random(X1.shape[1])
    w = np.zeros(X1.shape[1])
    final_iter = epochs

    for epoch in range(epochs):
        # print("\n")
        # print("epoch: {} {}".format(epoch, '-'*30))

        misclassified = 0
        for i, x in enumerate(X1):
            y = Y[i]
            h = np.dot(x, w)
            if h > 0:
                yhat = 1.
            else:
                yhat = 0.

            # if h <= 0:
            w = w - eta*(yhat-y)*x
            # misclassified += 1
            # print('misclassified? yes  w: {} '.format(w,i))

            # else:
            #     # print('misclassified? no  w: {}'.format(w))
            #     pass

        # if misclassified == 0:
        #     final_iter = epoch
        #     break

    return w, final_iter


def gen_lin_separable_data(data, data_tr, data_ts, data_size):
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, size=int(data_size/2))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, size=int(data_size/2))
    y2 = np.zeros(len(X1))#np.ones(len(X2)) * -1

    with open(data, 'w') as fo, \
            open(data_tr, 'w') as fo1, \
            open(data_ts, 'w') as fo2:
        for i in range(len(X1)):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X2[i][0], X2[i][1], y2[i])
            fo.write(line)
            fo.write(line2)

        for i in range(len(X1) - 20):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X2[i][0], X2[i][1], y2[i])
            fo1.write(line)
            fo1.write(line2)

        for i in range((len(X1) - 20), len(X1)):
            line = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X1[i][0], X1[i][1], y1[i])
            line2 = '{:5.2f} {:5.2f} {:5.0f} \n'.format(
                X2[i][0], X2[i][1], y2[i])
            fo2.write(line)
            fo2.write(line2)


def main():
    """Run main function."""

    # generate linearly separable data
    data = 'data.txt'
    data_tr = 'data_train.txt'
    data_ts = 'data_test.txt'
    data_size = 200
    gen_lin_separable_data(data, data_tr, data_ts, data_size)

    # read data
    X_train, Y_train = read_data(data_tr)
    X_test, Y_test = read_data(data_ts)

    # w = np.zeros(3)
    # plot_boundary(X_test, Y_test, w, 0)

    # fit perceptron
    epochs = 20
    eta = 0.1
    w, final_iter = perceptron_sgd(X_train, Y_train, epochs, eta)
    print('w = ', w)

    plot_boundary(X_test, Y_test, w, final_iter)

    # contour plot
    mesh_stepsize = 0.01
    plot_contour(X_test,Y_test,w,mesh_stepsize)

    from sklearn.linear_model import Perceptron
    ones = np.ones(X_train.shape[0]).reshape(X_train.shape[0], 1)
    x_train = np.append(ones, X_train, axis=1)
    y_train = Y_train
    clf = Perceptron(random_state=None, eta0=0.1,
                     shuffle=False, fit_intercept=False)
    clf.fit(x_train, y_train)
    # y_predict = clf.predict(X_test)

    print("sklearn weights:")
    print(clf.coef_[0])


if __name__ == "__main__":
    main()
