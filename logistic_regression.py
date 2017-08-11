# _*_ coding:utf-8 _*_

import numpy as np
import sklearn as sk
import random
import scipy as sp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#############################
##在进行疝气病预测的时候#####
##使用了21个特征！但实际上###
##我们并没有对这些特征进行预#
# #处理，这之中导致很多问题##
#############################
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


def predict(X_Test,Y_Test,theata):
    h = X_Test.dot(theata);
    Y_Pre = sigmod(h);
    ls = logloss(Y_Test,Y_Pre);
    return Y_Pre,ls;

def gradAscent(X_Train,Y_Train,theata,alpha,n_iter):
    m = X_Train.shape[0];
    J = np.zeros(n_iter).reshape(n_iter,1);
    for i in range(n_iter):
        index = i%(m-1);
        x =  X_Train[index,:].reshape(1,X_Train.shape[1]);
        y =  Y_Train[index,0].reshape(1,1);
        J[i,0] = predict(X_Train,Y_Train,theata)[1];
        h = x.dot(theata);
        delta = x.T.dot(y-sigmod(h));#x.T*y-x.T*h/(1+h);
        theata = theata + alpha*delta;
    return theata,J;

def loadDataSet():
    dataMat = [];
    fr = open('horse-colic-train.txt');
    for line in fr.readlines():
        lineArr = line.strip().split();
        dataMat.append([ float(lineArr[0]), float(lineArr[1]), 1.0, int(lineArr[2])]);
    return dataMat;

def loadHorseColic():
    Train_data = [];
    Test_data = [];
    frtrain = open('horseColicTraining.txt');
    frtest = open('horseColicTest.txt');
    for line in frtrain.readlines():
        lineArr = line.strip().split('\t');
        curline = [1];
        for i in range(len(lineArr)):
            curline.append(float(lineArr[i]));
        Train_data.append(curline);

    for line in frtest.readlines():
        lineArr = line.strip().split('\t');
        curline = [1];
        for i in range(len(lineArr)):
            curline.append(float(lineArr[i]));
        Test_data.append(curline);

    Train_data = np.array(Train_data);
    Test_data = np.array(Test_data);
    return Train_data, Test_data;
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

def classifyVector(inX, weights):
    prob = sigmod(np.sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def accutate_rate(theata):
    trainWeights = theata;
    frTest = open('horseColicTest.txt');
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if (int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21])):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is: %f' % errorRate)

# def stocGradAscent0(dataMatrix, classLabels):
#     dataMatrix = np.array(dataMatrix)
#     m,n = np.shape(dataMatrix)
#     alpha = 0.1
#     weights = np.ones(n)
#     for i in range(m):
#         h = sigmod(sum(dataMatrix[i] * weights))
#         error = classLabels[i] - h
#         weights = weights + alpha * error * dataMatrix[i]
#     return weights
#
# def stocGradAscent1(dataMatrix, classLabels, numInter = 150):
#     dataMatrix = np.array(dataMatrix)
#     m,n = np.shape(dataMatrix);
#     weights = np.ones(n);
#     for j in range(numInter):
#         dataIndex = range(m)
#         for i in range(m):
#             alpha = 4 / (1.0+j+i) + 0.01    #alpha值每次迭代时都进行调整
#             randIndex = int(random.uniform(0, len(dataIndex)))            #随机选取更新
#             h = sigmod(sum(dataMatrix[randIndex] * weights))
#             error = classLabels[randIndex] - h
#             weights = weights + alpha * error * dataMatrix[randIndex]
#             del[dataIndex[randIndex]]
#     return weights



if __name__ == '__main__':
    # data = loadDataSet();
    [Train,Test] = loadHorseColic();
    # data = [];
    # data = np.array(data);
    # n_samples = int(data.shape[0]*2/3);
    # X_Train = np.c_[data[:n_samples, 0],data[:n_samples, 1],data[:n_samples, 2]];  # load the X data, and merge b.
    # Y_Train = np.c_[data[:n_samples, 3]];  # load the y data
    #
    # X_Test = np.c_[data[n_samples:data.shape[0], 0], data[n_samples:data.shape[0], 1], data[n_samples:data.shape[0],2]];
    # Y_Test = np.c_[data[n_samples:data.shape[0], 3]];
    n_feat = Train.shape[1]-1;
    X_Train = Train[:,:n_feat];
    Y_Train = Train[:,n_feat];
    Y_Train = Y_Train.reshape(Y_Train.shape[0],1);

    X_Test = Test[:,:n_feat];
    Y_Test = Test[:,n_feat];
    Y_Test = Y_Test.reshape(Y_Test.shape[0],1);

    theata = np.ones(n_feat).reshape(n_feat,1);
    alpha = 0.001;
    n_iter  = 1500;
    [theata, J] = gradAscent(X_Train, Y_Train, theata, alpha, n_iter)
    [Y_Pre,loss] = predict(X_Test,Y_Test,theata);

    lr = LogisticRegression(solver='sag',max_iter=10000,fit_intercept=True,intercept_scaling = 1);
    lr.fit(X_Train[:,1:], Y_Train[:,0]);
    [pred_test, loss1] = predict(X_Test[:, :], Y_Test, np.c_[lr.intercept_, lr.coef_].T);

    accutate_rate(theata);
    accutate_rate(np.c_[lr.intercept_, lr.coef_].T);
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

