# _*_ coding:utf-8 _*_
import numpy as np
from numpy.matlib import repmat
import pandas as pd
import scipy as sp
from sklearn import linear_model


import matplotlib.pyplot as plt

##产生samples
def produce_data(filepath='linear_regression.txt'):
    N = 10000;
    x = np.linspace(1, 100, N);
    noise = (np.random.random(N)*2-1)*1e-1;
    y = np.sin(x)+noise; #add some noise.
    np.savetxt(filepath,np.stack((x,y)).transpose());


#带高斯核和L2正则的最小二乘（批梯度下降方法）
def computeCost(X,y,t,theta,w,lamda=0):
     m = X.shape[0];

     L2 = lamda*(1.0/(2*m))*np.sum(np.square(theta));
     J = (1.0/(2*m))*np.sum(np.square(w.T.dot(theta)-y))+L2;

     # J = (1.0 / (2 * m)) * np.sum(np.square(w.T.dot(theta) - y));
     # print(J);
     return J;

def gradientdecent(X, y, theta, t, alpha=0.001, num_iters=15000, lamda=0):
    m = X.shape[0];

    k = repmat(X[:, 1].T, m, 1);
    w = np.ones([m, m])*1.0;
    for i in range(m):
        w[:, i] = np.exp(-np.square(X[:, 1] - k[:, i])/ (2 * t ** 2)) ;

    J_history = np.zeros(num_iters);
    for i in range(num_iters):
        J_history[i] = computeCost(X, y, t, theta, w, lamda);
        L2 = (lamda*1.0/m)*(np.r_[theta[0:m]].reshape(m,1)*1.0);
        thetadelta = (1.0/m)*(w.dot((w.T.dot(theta)-y)))+L2;
        # thetadelta = (1.0 / m) * (w.dot((w.T.dot(theta) - y)));
        theta= theta - alpha*thetadelta;

    return w,theta,J_history;

def predict(x_data,X,Y,theta,t):
    m = x_data.shape[0];
    K = repmat(X[:, 1].T, m, 1);
    W = np.ones([m, X.shape[0]]);

    for i in range(m):
        W[:, i] = (np.exp(-np.square(x_data[:, 1] - K[:, i])/ (2 * t ** 2)));
    y_pre = W.T.dot(theta);
    return y_pre;

def loss(Y_Test,Y_pre):
    m = Y_Test.shape[0];
    J = (1.0 / (2 * m)) * np.sum(np.square(Y_pre - Y_Test));
    return J;


if __name__ == '__main__':
    produce_data();
    data = np.loadtxt('linear_regression.txt',delimiter=' ');

    n_samples = 5000;
    X_Train = np.c_[np.ones(n_samples),data[:n_samples,0]]; #load the X data, and merge b.
    Y_Train = np.c_[data[:n_samples,1]];                    #load the y data

    X_Test = np.c_[np.ones(5000),data[5000:10000,0]];
    Y_Test = np.c_[data[5000:10000,1]]

    alpha = 0.001;
    lamda = 0.01;
    t = 0.3;
    Iter = 1500;
    theta = np.zeros([X_Train.shape[0],1]);

    [W,theta,Cost_J] = gradientdecent(X_Train,Y_Train, alpha=alpha, num_iters=Iter,theta=theta,t=t,lamda=lamda);
    Y_pre = predict(X_Train,X_Test,Y_Test, theta=theta, t=t);

    #the tendency line
    # plt.xlim(0,1500);
    # plt.ylim(0,3);
    # plt.plot(Cost_J);
    # plt.show();

    # Create linear regression object with regular.
    [intercept, coef] = linear_model.ridge_regression(X_Train[:,1].reshape(-1,1), Y_Train,alpha=0.001,max_iter=15000,solver='sag',return_intercept=True);#linear_model.LinearRegression()
    # Train the model using the training sets
    #regr.fit(X_Train[:,1].reshape(-1,1), Y_Train);
    #predict the test data
    result = X_Test.dot(np.r_[coef.reshape(-1,1),intercept.reshape(1,1)].reshape(2,1));#= regr.predict(X_Test);

    print(loss(Y_Test,result));
    print(loss(Y_Test,Y_pre));
    #########################
    # how to draw the scater#
    #########################
    plt.scatter(X_Test[:,1],Y_pre,s=30,c='r',marker='x',linewidths=1);
    plt.xlim(70,80);
    # plt.ylim(-1,1);
    plt.xlabel('x');
    plt.ylabel('y');
    plt.show();
