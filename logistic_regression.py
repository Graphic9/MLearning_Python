# _*_ coding:utf-8 _*_

import numpy as np
import sklearn as sk
import random
import scipy as sp
from sklearn.linear_model import LogisticRegression
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
  ll = ll * 1.0/len(act);
  print("the loss is %lf!" % -ll);
  return -ll;

#对数损失函数
# def computCost(X_Train,Y_Train,theata):
#     m = X_Train.shape[0];
#     h = np.exp(X_Train*theata);
#     J = Y_Train.T*h-np.sum(np.log(1+h));
#     print("Error is : %lf" % -J);
#     return -J;

def predict(X_Test,Y_Test,theata):
    h = X_Test.dot(theata);
    Y_Pre = np.mat(sigmod(h));
    ls = logloss(Y_Test,Y_Pre);
    return Y_Pre,ls;

def gradAscent(X_Train,Y_Train,theata,alpha,n_iter):
    m = X_Train.shape[0];
    J = np.zeros(n_iter).reshape(n_iter,1);
    for i in range(n_iter):
        index = i%(m-1);
        x = X_Train[index,:].reshape(1,X_Train.shape[1]);
        y = Y_Train[index,0].reshape(1,1);
        J[i,0] = predict(X_Train,Y_Train,theata)[1];
        h = x.dot(theata);
        delta = x.T.dot((y-sigmod(h)));#x.T*y-x.T*h/(1+h);
        theata = theata + alpha*delta;
    return theata,J;

def loadDataSet():
    dataMat = [];
    fr = open('testSet.txt');
    for line in fr.readlines():
        lineArr = line.strip().split();
        dataMat.append([ float(lineArr[0]), float(lineArr[1]), 1.0, int(lineArr[2])]);
        # labelMat.append()
    return dataMat;


# 分析数据，画出决策边界
def plotBestFit(wei, dataMatrix, labelMat):
    weights = wei;  # 将矩阵wei转化为list
    dataArr = dataMatrix;  # 将矩阵转化为数组
    n = dataMatrix.shape[0];
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) >= 0.5:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[2] - weights[0] * x) / weights[1]
    ax.plot(x, y)
    plt.xlabel("x1")  # X轴的标签
    plt.ylabel("x2")  # Y轴的标签
    plt.show()



if __name__ == '__main__':
    #produce_data();
    # data = np.loadtxt('logistic_regression.txt', delimiter='    ');
    data = loadDataSet();

    data = np.array(data);
    n_samples = int(data.shape[0]*2/3);
    X_Train = np.c_[data[:n_samples, 0],data[:n_samples, 1],data[:n_samples, 2]];  # load the X data, and merge b.
    Y_Train = np.c_[data[:n_samples, 3]];  # load the y data

    X_Test = np.c_[data[n_samples:data.shape[0], 0], data[n_samples:data.shape[0], 1], data[n_samples:data.shape[0],2]];
    Y_Test = np.c_[data[n_samples:data.shape[0], 3]];

    theata = np.zeros(X_Train.shape[1]).reshape(3,1);
    alpha = 0.001;
    n_iter  = 1000000;
    [theata, J] = gradAscent(X_Train, Y_Train, theata, alpha, n_iter)
    [Y_Pre,loss] = predict(X_Test,Y_Test,theata);

    lr = LogisticRegression(solver='sag',max_iter=10000,fit_intercept=True,intercept_scaling = 1);
    lr.fit(X_Train[:,0:2], Y_Train[:,0]);
    # pred_test = lr.predict(X_Test[:,0:2]).reshape(34,1);

    plotBestFit(theata, X_Test, Y_Test);

    [pred_test,loss1] = predict(X_Test,Y_Test,np.c_[lr.coef_,lr.intercept_].T);
    print('the last loss is %lf' % loss);
    print('the last loss is %lf' % loss1);
    print('the max_iterator is %d' % lr.max_iter);

    # plt.xlim(-10,10);
    # plt.ylim(-1,1);
    # plt.xlabel('value of x');
    # plt.ylabel('value of y');
    # plt.plot(x,y,'g');
    # plt.show();
    # print('hello world!')

