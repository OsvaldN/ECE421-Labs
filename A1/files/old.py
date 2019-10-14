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
    grad_bias = np.sum(np.matmul(x, W) + b - y, keepdims=True)/len(y)

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

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    plt_weights = np.sum(W)/len(W)
    plt_train_err = MSE(W, b, trainingData, trainingLabels, reg)
    W_mag_new = np.linalg.norm(W)
    for i in range(iterations):
        grad_w, grad_b = gradMSE(W, b, trainingData, trainingLabels, reg)
        W = W - alpha*grad_w
        b = b - alpha*grad_b
        W_mag_old = W_mag_new
        W_mag_new = np.linalg.norm(W)
        Err = (W_mag_new - W_mag_old)/W_mag_new
        if Err < EPS:
            break
        #add weights for plots
        if i % 10 == 0:
            plt_weights = np.append(plt_weights, np.sum(W)/len(W))
            plt_train_err = np.append(plt_train_err, MSE(W, b, trainingData, trainingLabels, reg))
    plt.plot(plt_weights, label='Weights')
    plt.plot(plt_train_err, label='training')
    plt.title("Training Curve")
    plt.xlabel("Iterations")
    plt.legend(loc='best')
    plt.show()
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass

def linear_normal(x, y, reg):
    #closed form linear regression least squares calculation
    W = np.zeros((x.shape[1], 1))
    W = np.insert(W, 0, 0)
    x = np.insert(x, [0], 1, axis=1)

    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
    return W[1:], W[0]

def accuracy(W, b, x, y):
    '''
    returns fraction of correct predictions of Wx + b = y
    '''
    y_hat = np.matmul(x, W) + b
    num_correct = 0
    #np.apply_along_axis(lambda x: np.round(x), 0, y_hat)
    y_hat = np.around(y_hat)
    y_hat = y_hat.astype(bool)
    err = y - y_hat
    return (1 - (np.count_nonzero(err) / len(err))) * 100

#[trainData, validData, testData, trainTarget, validTarget, testTarget]
dataList = loadData()
dataList = list(dataList)


# Data stored as (sample size (N), flattened image (d))
for i, data in enumerate(dataList[:3]):
    dataList[i] = data.reshape(len(data), -1)

# Initialize weights
weights = np.zeros((dataList[0].shape[1], 1))
bias = np.zeros((1, 1))

trained_w, trained_b = grad_descent(weights, bias, dataList[0], dataList[3], 5e-3, 50, 5e-3, 1e-7)
norm_w, norm_b = linear_normal(dataList[0], dataList[3], 0)
print(accuracy(trained_w, dataList[0], trained_b, dataList[3]))
print(accuracy(norm_w, dataList[0], norm_b, dataList[3]))