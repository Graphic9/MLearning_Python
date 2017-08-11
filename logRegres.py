# -*- coding: cp936 -*-
from numpy import *

#this is the code downloading from:http://blog.csdn.net/lu597203933/article/details/38468303


#加载数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#计算sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法-计算回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)          #转换为numpy数据类型
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#画出决策边界
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else: xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]- weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.1
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numInter = 150):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numInter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01    #alpha值每次迭代时都进行调整
            randIndex = int(random.uniform(0, len(dataIndex)))            #随机选取更新
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # delete(randIndex, (0), axis=0);
            # del[dataIndex[randIndex]]
    return weights


#案例-从疝气病症预测病马的死亡率
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    # trainWeights = stocGradAscent0(trainingSet, trainingLabels);
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests, errorSum/float(numTests)))
    

if __name__ == "__main__":
    # multiTest();
    [dataMatrix, classLabels] = loadDataSet();
    # stocGradAscent1(dataMatrix, classLabels, numInter=150)
    colicTest();
