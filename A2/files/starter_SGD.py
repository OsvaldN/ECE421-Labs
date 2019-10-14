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
    y_hat = np.matmul(X, W) + b
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
    # hidden layer
    z_h = computeLayer(x, W_h, b_h)
    x_h = relu(z_h)
    # output layer
    z_o = computeLayer(x_h, W_o, b_o)
    x_o = softmax(z_o)

    return x_o, x_h

def backProp(target, x_o, x_h, x, W_o):
    ''' computes gradients wrt weights and biases
     '''

    #dL = gradCE(target, x_o) # (N, 10, 1) array
    #dx_o = gradSoftmax(x_o) # (N, 10,10) array

    dL_dz_o = target - x_o # (N, 10) array

    db_o = np.expand_dims(dL_dz_o, axis=2) # (N, 10, 1) array
    dW_o = np.matmul(x_h.reshape(len(x_h),-1,1), np.transpose(db_o,axes=(0,2,1))) # (N, k, 10) array

    W_o = np.array([W_o] * len(db_o))
    theta_p = np.expand_dims(np.array(x_h>0, dtype=int), axis = 2)
    db_h = np.multiply(np.matmul(W_o, db_o), theta_p) # (N, K, 1) array
    dW_h = np.matmul(np.expand_dims(x, axis=2), np.transpose(db_h, axes=(0,2,1))) # (N, 784, 10) array

    db_o = np.sum(db_o, axis=0)/len(db_o)
    dW_o = np.sum(dW_o, axis=0)/len(dW_o)
    db_h = np.sum(db_h, axis=0)/len(db_h)
    dW_h = np.sum(dW_h, axis=0)/len(dW_h)

    return db_o.T, dW_o, db_h.T, dW_h
    

def gradSoftmax(x_o):
    ''' gradient of softmax outputs wrt inputs
     x_o: (N, C) matrix of predictions
     '''
    out = np.zeros((x_o.shape[0],x_o.shape[1],x_o.shape[1]))
    for i,ex in enumerate(x_o):
        jac = np.zeros(len(ex),len(ex))
        for i in range(len(jac)):
            for j in range(len(jac)):
                if i == j:
                    jac[i][j] = ex[i] * (1-ex[j])
                else: 
                    jac[i][j] = -ex[i] * ex[j]
        out[i] = jac
    return out

def XavierInit(into, out):
    ''' Xavier initialization for weight matrix '''
    return np.random.normal(0,2/(into+out), (into, out))


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
# flatten images
trainData = trainData.reshape(len(trainData), -1,)[0:40]
validData = validData.reshape(len(validData), -1)[0:40]
testData = testData.reshape(len(testData), -1)[0:40]

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

train_accuracy = []
valid_accuracy = []
test_accuracy = []
train_loss = []
valid_loss = []
test_loss = []
sum_wh =[]
sum_wo =[]
epochs = 200
for epoch in range(epochs):
    train_l= 0
    valid_l = 0
    test_l = 0
    trainData, trainTarget = shuffle(trainData, trainTarget)
    correct = 0
    total = 0
    for b in range(len(trainData)//4):
        x_o, x_h = forwardProp(trainData[(4*b):4*(b+1)], W_h, b_h, W_o, b_o)
        db_o, dW_o, db_h, dW_h = backProp(trainTarget[(4*b):4*(b+1)], x_o, x_h, trainData[(4*b):4*(b+1)], W_o)
        #update momentums
        vW_h = (0.99 * vW_h) + (0.00001 * dW_h)
        vb_h = (0.99 * vb_h) + (0.00001 * db_h)
        vW_o = (0.99 * vW_o) + (0.00001 * dW_o)
        vb_o = (0.99 * vb_o) + (0.00001 * db_o)
        # Descend Gradients
        W_h = W_h + vW_h
        b_h = b_h + vb_h
        W_o = W_o + vW_o
        b_o = b_o + vb_o
        
        train_l += CE(trainTarget[(4*b):4*(b+1)], x_o)
        correct += np.sum((np.argmax(x_o, axis=1) == np.argmax(trainTarget[(4*b):4*(b+1)], axis=1)))
        total += 4

    train_loss.append(train_l)
    train_accuracy.append(correct/total)
    correct,total = 0,0

    for b in range(len(testData)//4):
        x_o, x_h = forwardProp(testData[(4*b):4*(b+1)], W_h, b_h, W_o, b_o)

        correct += np.sum((np.argmax(x_o, axis=1) == np.argmax(testTarget[(4*b):4*(b+1)], axis=1)))
        total += 4
        test_l += CE(testTarget[(4*b):4*(b+1)], x_o)
    test_loss.append(test_l)
    test_accuracy.append(correct/total)
    correct,total = 0,0

    for b in range(len(validData)//4):
        x_o, x_h = forwardProp(validData[(4*b):4*(b+1)], W_h, b_h, W_o, b_o)

        valid_l += CE(validTarget[(4*b):4*(b+1)], x_o)
        correct += np.sum((np.argmax(x_o, axis=1) == np.argmax(validTarget[(4*b):4*(b+1)], axis=1)))
        total += 4
    valid_loss.append(valid_l)
    valid_accuracy.append(correct/total)
    
    if epoch % 10 == 0:
        print('epoch %d complete' % (epoch+1))
        print(train_accuracy[-1])

plt.plot(test_loss, 'r', label='test')
plt.plot(train_loss, 'b', label='train')
plt.plot(valid_loss, 'g', label='validation')
plt.title('training accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()