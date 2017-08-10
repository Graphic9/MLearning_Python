# _*_ coding:utf-8 _*_

import numpy as np
import sklearn as sk
import random
import scipy as sp
import matplotlib.pyplot as plt

def produce_data(filepath='logistic_regression.txt'):
    N = 10000;
    x = np.linspace(1, 100, N);
    y = np.random.random(N)*100;
    label = np.round(np.random.random(N));
    np.savetxt(filepath,np.stack((x,y,label)).transpose());

def sigmod(x):
    y = 1.0/(1+np.exp(-x));
    return y;

def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(np.multiply(act,sp.log(pred)) + np.multiply(sp.subtract(1,act),sp.log(sp.subtract(1,pred))))
  ll = ll * 1.0/len(act)
  return -ll;

#对数损失函数
# def computCost(X_Train,Y_Train,theata):
#     m = X_Train.shape[0];
#     h = np.exp(X_Train*theata);
#     J = Y_Train.T*h-np.sum(np.log(1+h));
#     print("Error is : %lf" % -J);
#     return -J;

def predict(X_Test,Y_Test,theata):
    h = X_Test*theata;
    Y_Pre = np.mat(sigmod(h));
    ls = logloss(Y_Test,Y_Pre);
    return Y_Pre,ls;

def gradAscent(X_Train,Y_Train,theata,alpha,n_iter):
    m = X_Train.shape[0];
    J = np.mat(np.zeros(n_iter).reshape(n_iter,1));
    for i in range(n_iter):
        index = int(np.round(np.random.random(1)[0]*m));
        x = X_Train[index,:];
        y = Y_Train[index,0];
        J[i,0] = predict(X_Train,Y_Train,theata)[1];
        h = x.dot(theata);
        delta = x.T.dot((y-sigmod(h)));#x.T*y-x.T*h/(1+h);
        theata = theata + alpha*delta;
    return theata,J;


if __name__ == '__main__':
    # produce_data();
    data = np.loadtxt('logistic_regression.txt', delimiter=' ');

    n_samples = 5000;
    X_Train = np.mat(np.c_[data[:n_samples, 0],data[:n_samples, 1],np.ones(n_samples)]);  # load the X data, and merge b.
    Y_Train = np.mat(np.c_[data[:n_samples, 2]]);  # load the y data

    X_Test = np.mat(np.c_[np.ones(5000), data[5000:10000, 0], data[5000:10000, 1]]);
    Y_Test = np.mat(np.c_[data[5000:10000, 2]]);

    theata = np.mat(np.zeros(X_Train.shape[1]).reshape(3,1));
    alpha = 0.01;
    n_iter  = 150;
    [theata, J] = gradAscent(X_Train, Y_Train, theata, alpha, n_iter)
    [Y_Pre,loss] = predict(X_Test,Y_Test,theata);

    print('the last loss is %lf' % loss);

    # plt.xlim(-10,10);
    # plt.ylim(-1,1);
    # plt.xlabel('value of x');
    # plt.ylabel('value of y');
    # plt.plot(x,y,'g');
    # plt.show();
    # print('hello world!')

