import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    #return max(0,x) element-wise on array x
    return x.clip(min=0)

def softmax(x):
    #vectorized softmax
    # x: (N, C) array of N examples and C classes
    # theta: (N, C) softmax activations for each example/class
    numerator = np.exp(x)
    denominator = np.sum(np.exp(x), axis=1, keepdims=True)
    theta = numerator/denominator
    return theta

def computeLayer(X, W, b):
    # X: (N, dim(l-1)) array of N examples
    # W: (dim(l-1), dim(l)) array of weights
    # b: (1, dim(l)) bias term 
    # y_hat: (N, dim(l)) array output
    print('X'+str(X.shape))
    print('W'+str(W.shape))
    print('b'+str(b.shape))
    y_hat = np.matmul(X, W) + b
    print('y_hat'+str(y_hat.shape))
    return y_hat

def CE(target, prediction):
    # return average cross entropy loss
    # target: (N, K) one-hot encoding of labels
    # prediction (N, K) probability outputs
    N = target.shape[0]
    loss = -np.sum(np.multiply(target, np.log(prediction))) / N
    return loss

def gradCE(target, prediction):
    # return total cross entropy loss
    # target: (N, K) one-hot encoding of labels
    # prediction (N, K) probability outputs
    N = len(target)
    grad = np.divide(target, np.log(prediction)).reshape(N, -1, 1)
    return grad

def forwardProp(x, W_h, b_h, W_o, b_o):
    ''' computes output while storing cache for backProp
     x: (N, 784) input vector passed through K hidden units
     W_h, W_o: (784, K), (K, 10) weight matrices at hidden and output layers
     b_h, b_o: (1, K), (1, 10) bias vector at hidden and output layers
     return (N, 10) prediction (N, K) hidden layer activations for backProp '''
    x = np.expand_dims(x, axis=1)
    # hidden layer
    z_h = computeLayer(x, W_h, b_h)
    x_h = relu(z_h)
    # output layer
    z_o = computeLayer(x_h, W_o, b_o)
    x_o = softmax(z_o)
    x_o = np.squeeze(x_o, axis=1)
    return x_o, x_h

def backProp(target, x_o, x_h, x, W_o):
    ''' computes gradients wrt weights and biases
     '''
    print('x_o'+str(x_o.shape))
    dL = gradCE(target, x_o) # (N, 10, 1) array
    dx_o = gradSoftmax(x_o) # (N, 10,10) array

    db_o = np.transpose(np.matmul(np.transpose(dL,(0,2,1)), np.transpose(dx_o, (0,2,1))), axes=(0,2,1)) # (N, 10, 1) array
    dW_o = np.matmul(x_h.reshape(len(x_h),-1,1), np.transpose(db_o,axes=(0,2,1))) # (N, k, 10) array

    W_o = W_o.reshape(1, W_o.shape[0], W_o.shape[1])
    theta_p = np.expand_dims(np.array(x_h>0, dtype=int), axis = 2)
    db_h = np.multiply(np.matmul(W_o, db_o), theta_p) # (N, K, 1) array
    print('db_h'+str(db_h.shape))
    print('x'+str(x.shape))
    dW_h = np.matmul(np.expand_dims(x, axis=2), np.transpose(db_h, axes=(0,2,1))) # (N, 784, 10) array

    return db_o, dW_o, db_h, dW_h
    

def gradSoftmax(x_o):
    ''' gradient of softmax outputs wrt inputs
     x_o: (N,1, C) matrix of predictions
     '''

    out = np.zeros((x_o.shape[0],x_o.shape[1],x_o.shape[1]))
    for i,ex in enumerate(x_o):
        s = ex.reshape(-1,1)
        out[i] = np.diagflat(s) - np.dot(s, s.T)
    return out

def XavierInit(into, out):
    ''' Xavier initialization for weight matrix '''
    return np.random.normal(0,2/(into+out), (into, out))


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
# flatten images
trainData = trainData.reshape(len(trainData), -1,)
validData = validData.reshape(len(validData), -1)
testData = testData.reshape(len(testData), -1)

''' __net__ '''
K=1000
W_h = XavierInit(784, K)
b_h = np.zeros((1,K))
W_o = XavierInit(K, 10)
b_o = np.zeros((1, 10))

vW_h = 1e-5 * np.ones((784, K))
vb_h = 1e-5 * np.ones((1, K))
vW_o = 1e-5 * np.ones((K, 10))
vb_o = 1e-5 * np.ones((1, 10))

for epoch in range(3):
    for b in range(len(trainData)//4):
        x_o, x_h = forwardProp(trainData[(4*b):4*(b+1)], W_h, b_h, W_o, b_o)
        db_o, dW_o, db_h, dW_h = backProp(trainTarget[(4*b):4*(b+1)], x_o, x_h, trainData[(4*b):4*(b+1)], W_o)
        #update momentums
        vW_h = (0.9 * vW_h) + (0.1 * dW_h)
        vb_h = (0.9 * vb_h) + (0.1 * db_h)
        vW_o = (0.9 * vW_o) + (0.1 * dW_o)
        vb_o = (0.9 * vb_o) + (0.1 * db_o)
        # Descend Gradients
        W_h = W_h - vW_h
        b_h = b_h - vb_h
        W_o = W_o - vW_o
        b_o = b_o - vb_o