import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    y_hat = np.matmul(x, W) + b

    err = np.square(y_hat - y)
    loss_mse = np.sum(err)/(2*len(y))
    loss_reg = (np.linalg.norm(W)**2)*reg/2
    return loss_mse + loss_reg

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    #W = np.insert(W, 0, 1)
    #x = np.insert(x, [0], b, axis=1)

    # grad w.r.t. weights
    grad_mse = ((np.matmul(np.matmul(x.T, x), W)).reshape(len(W), 1) \
                - np.matmul(x.T, y))/len(y)
    grad_reg = reg * np.linalg.norm(W)

    # grad w.r.t. bias
    grad_bias = np.sum(np.matmul(x, W) + b - y, keepdims=True)

    # return tuple of grad w.r.t. weights and bias
    return (grad_mse + grad_reg), grad_bias

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    z = (np.matmul(x, W) + b)
    y_hat = 1/(1 + np.exp(-z))

    err = -y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)
    loss_ce = np.sum(err) / len(y)
    loss_reg = (np.linalg.norm(W)**2)*reg/2
    return loss_ce + loss_reg

def gradCE(W, b, x, y, reg):
    # Your implementation here
    z = (np.matmul(x, W) + b)
    y_hat = 1/(1 + np.exp(-z))
    y_hat = y_hat.reshape((len(y_hat), -1))

    # grad w.r.t. weights
    grad_l = (-y/y_hat + (1-y)/(1-y_hat))
    grad_sig = np.matmul((y_hat * (1 - y_hat)).T, x)
    grad_reg = reg * np.linalg.norm(W)
    grad_w = np.sum(grad_l * grad_sig, axis=0, keepdims=True).T / len(y) + grad_reg

    # grad w.r.t. bias
    grad_b = np.sum(grad_l * (y_hat * (1 - y_hat)), keepdims=True) / len(y)

    # return tuple of grad w.r.t. weights and bias
    return grad_w, grad_b

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, otherData, EPS=1e-7):
    # Maybe initialize weights here?
    plt_weights = np.sum(W)/len(W)
    plt_train_err = MSE(W, b, trainingData, trainingLabels, reg)
    plt_val_err = MSE(W, b, otherData[1], otherData[4], reg)
    while(iterations > 0):
        g_t = gradMSE(W, b, trainingData, trainingLabels, reg)

        # if avg of grad. is smaller than error tolerance, end descent
        if(alpha * np.abs(np.sum(g_t[0])/len(g_t)) < EPS): #note, should maybe make this MSE not gradient
            break

        dir_t = -g_t[0]
        #dir_b = -g_t[1]
        W += alpha * dir_t
        #b += alpha * dir_b
        #TODO: save values for plotting
        if iterations % 10 == 0:
            plt_weights = np.append(plt_weights, np.sum(W)/len(W))
            plt_train_err = np.append(plt_train_err, MSE(W, b, trainingData, trainingLabels, reg))
            plt_val_err = np.append(plt_val_err, MSE(W, b, otherData[1], otherData[4], reg))

        iterations -= 1

    # plot data
    plt.plot(plt_weights, label='Weights')
    plt.plot(plt_train_err, label='training')
    plt.plot(plt_val_err, label='validation')
    plt.title("Training Curve")
    plt.xlabel("Iterations")
    plt.legend(loc='best')
    plt.show()
    return W, b



def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass

#[trainData, validData, testData, trainTarget, validTarget, testTarget]
dataList = loadData()
dataList = list(dataList)


# Data stored as (sample size (N), flattened image (d))
for i, data in enumerate(dataList[:3]):
    dataList[i] = data.reshape(len(data), -1)

# Initialize weights
weights = np.zeros((dataList[0].shape[1], 1))
bias = np.zeros((1, 1))

'''
loss = MSE(weights, bias, dataList[0], dataList[3], 0)
print(loss)

loss = gradMSE(weights, bias, dataList[0], dataList[3], 0)
print(loss[0].shape)
print(loss[1].shape)

loss = crossEntropyLoss(weights, bias, dataList[0], dataList[3], 0)
print(loss)

loss = gradCE(weights, bias, dataList[0], dataList[3], 0)
print(loss[0].shape)
print(loss[1].shape)
'''

trained = grad_descent(weights, bias, dataList[0], dataList[3], 5e-3, 500, 0, dataList)
#print(trained[0], "\n", trained[1])
