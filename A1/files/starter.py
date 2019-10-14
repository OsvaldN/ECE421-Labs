import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time as time

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
    # grad w.r.t. weights
    grad_mse = ((np.matmul(np.matmul(x.T, x), W)).reshape(len(W), 1) \
                - np.matmul(x.T, y))/len(y)
    grad_reg = reg * W

    # grad w.r.t. bias
    grad_bias = np.sum(np.matmul(x, W) + b - y, keepdims=True) / len(y)

    # return tuple of grad w.r.t. weights and bias
    return (grad_mse + grad_reg), grad_bias

def sigmoid(W, b, x):
    #return sigmoid activation
    z = np.matmul(x, W) + b
    a = 1/(1 + np.exp(-z))
    return a

def linear(W, b, x):
    a = np.matmul(x, W) + b
    return a

def crossEntropyLoss(W, b, x, y, reg):
    y_hat = sigmoid(W, b, x)

    log_loss = -(y * np.log(y_hat)) - (1 - y) * np.log(1 - y_hat)
    loss = np.sum(log_loss) / len(y) + (np.linalg.norm(W)**2)*reg/2
    loss = np.squeeze(loss)
    assert(isinstance(loss, float))
    return loss

def gradCE(W, b, x, y, reg):
    y_hat = sigmoid(W, b, x)

    #grad_w = np.dot(x.T, y_hat - y) / len(y) + (reg * np.linalg.norm(W) / len(y))
    grad_w = np.dot(x.T, y_hat - y) / len(y) + (reg * W)
    grad_b = np.sum(y_hat - y, keepdims=True) / len(y)

    # return tuple of grad w.r.t. weights and bias
    return grad_w, grad_b

def linear_normal(x, y, reg):
    # returns optimal weights and bias for binary classification linear regression
    start = time.clock()
    x = np.insert(x, [0], 1, axis=1)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
    end = time.clock()
    print("normal: ", end-start)
    return W[1:], W[0]

### TODO: REMOVE VALIDATION DATA
### TODO: TRACK TRAINING TIME
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, otherData, EPS=1e-7, loss=MSE, grad=gradMSE):
    # otherData stores unseen validation dataset
    plt_train_err = loss(W, b, trainingData, trainingLabels, reg)
    plt_val_err = loss(W, b, otherData[1], otherData[4], reg)
    plt_test_err = loss(W, b, otherData[2], otherData[5], reg)

    W_new = np.copy(W)
    start = time.clock()
    for i in range(1, iterations):
        grad_w, grad_b = grad(W, b, trainingData, trainingLabels, reg)
        W -= alpha * grad_w
        b -= alpha * grad_b
        #print(np.sum(alpha * grad_w)/len(alpha * grad_w))

        # if change in weights is smaller than error tolerance, end descent
        W_old = np.copy(W_new)
        W_new = np.copy(W)
        if(np.sqrt(np.square(np.linalg.norm(W_new-W_old)) + np.square(b)) < EPS):
            print("hit error tolerance")
            break

        if i % 10 == 0:
            plt_train_err = np.append(plt_train_err, loss(W, b, trainingData, trainingLabels, reg))
            plt_val_err = np.append(plt_val_err, loss(W, b, otherData[1], otherData[4], reg))
            plt_test_err = np.append(plt_test_err, loss(W, b, otherData[2], otherData[5], reg))

    end = time.clock()
    print(end - start)

    # plot data
    plot(plt_train_err, plt_val_err, plt_test_err, accuracy(W, b, otherData[2], otherData[5], activation=sigmoid), alpha, reg, False)
    return W, b

def plot(training, validation, testing, test, alpha, reg, mse):
    '''
    makes a plot of the training and validation error at every 10 epochs through training, and compares them to the test accuracy.
    Includes learning rate, classification type and regularization parameters in title
    input:
        training: MSE/CE error of training data every 10 epochs
        validation: MSE/CE error of training data every 10 epochs
        test: final accuracy after training
        alpha: learning rate
        reg: regularization parameter
        mse: flag saying where it is linear or logistic regresion
    '''
    plt.figure()
    plt.plot(training, label='Training')
    plt.plot(validation, label='Validation')
    plt.plot(testing, label='Testing')

    # Formatting
    kind = 'Linear' if mse else 'Logistic'
    title = "Regression Training Error\nwith Learning Rate "+\
            "of %.4f and Regularization of %.3f" % (alpha, reg) #+\
            #"\nfinal test accuracy: " + "%.2f" % (test*100) + "%"

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.draw()
    return

def accuracy(W, b, x, y, activation=sigmoid):
    # returns fraction of correct predictions of Wx + b = y
    #y_hat = np.matmul(x, W) + b
    y_hat = activation(W, b, x)
    y_hat = np.around(y_hat)
    y_hat = y_hat.astype(bool)
    err = y - y_hat
    return (1 - (np.count_nonzero(err) / len(err))) #* 100


def buildGraph(beta1=None, beta2=0.9999, epsilon=None, lossType='MSE', alpha=1e-3):
    g = tf.Graph()
    tf.set_random_seed(421)

    # Init weight and bias tensors
        # weight tensors use tf.truncated_normal w/ std. dev. = 0.5
    weights = tf.get_variable("weights", [784, 1], initializer=tf.truncated_normal_initializer(stddev=0.5))
    bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))

    # use tf.placeholder to create tensors for data, labels, and reg
    data = tf.placeholder(tf.float32, name="data")
    labels = tf.placeholder(tf.uint8, name="labels")
    reg = tf.placeholder(tf.float32, name="reg")

    predictedLabels = tf.math.sigmoid(tf.matmul(data, weights) + bias)
    loss = tf.losses.sigmoid_cross_entropy(labels, predictedLabels)
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
    accuracy = tf.metrics.accuracy(labels, predictedLabels)

    return weights, bias, data, labels, loss, reg, optimizer, accuracy

def train_model(trainData, trainLabels, valData, valLabels, testData, testLabels,  weights, bias, data, labels, loss, reg, optimizer, accuracy, batch_size=500, num_epochs=700):
    with tf.Session() as sess:
        # TODO: add write to tensorboard for graph visualization
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(tf.local_variables_initializer())

        training_plot = np.array([])
        val_plot = np.array([])
        test_plot = np.array([])
        accuracy_plot = np.array([])
        for epoch in range(num_epochs):
            # shuffle and iterate through in sets of 500
            np.random.seed(421)
            randIndx = np.arange(len(trainData))
            np.random.shuffle(randIndx)
            trainData, trainLabels = trainData[randIndx], trainLabels[randIndx]
            for batch in range(int(len(dataList[0])/batch_size)):
                # Optimize loss function
                sess.run(optimizer, feed_dict={data : trainData[batch:batch + batch_size], labels : trainLabels[batch:batch + batch_size], reg : 0})

            train_err = sess.run(loss, feed_dict={data : trainData, labels : trainLabels, reg : 0})
            val_err = sess.run(loss, feed_dict={data : valData, labels : valLabels, reg : 0})
            test_err = sess.run(loss, feed_dict={data : testData, labels : testLabels, reg : 0})
            acc, _ = sess.run(accuracy, feed_dict={data : testData, labels : testLabels, reg : 0})

            training_plot = np.append(training_plot, train_err)
            val_plot = np.append(val_plot, val_err)
            test_plot = np.append(test_plot, test_err)
            accuracy_plot = np.append(accuracy_plot, acc)

        #plt.plot(training_plot, label='Train')
        #plt.plot(val_plot, label='Validation')
        #plt.plot(test_plot, label='Test')
        plot(training_plot, val_plot, test_plot, 0, 1e-3, 0, True)
        plt.figure()
        plt.plot(100 * accuracy_plot, label="accuracy")
        plt.show()



# Initialize weights
def initializeWB():
    W = np.zeros((dataList[0].shape[1], 1))
    b = np.zeros((1, 1))

    return W, b


#[trainData, validData, testData, trainTarget, validTarget, testTarget]
dataList = loadData()
dataList = list(dataList)

# Data stored as (sample size (N), flattened image (d))
for i, data in enumerate(dataList[:3]):
    dataList[i] = data.reshape(len(data), -1)

# Train tensorflow model
graph = buildGraph()
train_model(dataList[0], dataList[3], dataList[1], dataList[4], dataList[2], dataList[5], graph[0], graph[1], graph[2], graph[3], graph[4], graph[5], graph[6], graph[7])

'''
#Loops
rates = [5e-3, 1e-3, 1e-4] # for A1 use [5e-3, 1e-3, 1e-4]
regs = [0.001, 0.1, 0.5]
loss_func = crossEntropyLoss
#loss_func = MSE
grad_func = gradCE
#grad_func = gradMSE
for rate in [0.005]:
    weights, bias = initializeWB()
    trained_w, trained_b = grad_descent(weights, bias, dataList[0], dataList[3], rate, 5000, 0, dataList, loss=loss_func, grad=grad_func)
    print(accuracy(trained_w, trained_b, dataList[2], dataList[5], activation=sigmoid))
    #normal_w, normal_b = linear_normal(dataList[0], dataList[3], 0)
    #print(accuracy(normal_w, normal_b, dataList[2], dataList[5]))
plt.show()
weights, bias = initializeWB()
trained_w, trained_b = grad_descent(weights, bias, dataList[0], dataList[3], 0.005, 500, 0, dataList, loss=loss_func, grad=grad_func)
trained_w, trained_b = grad_descent(trained_w, trained_b, dataList[0], dataList[3], 0.005, 4500, 0, dataList, loss=loss_func, grad=grad_func)
plt.show()
weights, bias = initializeWB()
normal_w, normal_b = linear_normal(dataList[0], dataList[3], 0, )
print(accuracy(normal_w, normal_b, dataList[2], dataList[5], activation=linear))
'''