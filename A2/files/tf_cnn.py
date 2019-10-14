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
    randIndx = np.arange(int(trainData.shape[0]))
    print(trainTarget.shape)
    target = trainTarget
    print(trainTarget.shape)
    print(target.shape)
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    # TODO
    pass

def softmax(x):
    # TODO
    pass


def computeLayer(X, W, b):
    # TODO
    pass

def CE(target, prediction):
    # TODO
    pass

def gradCE(target, prediction):
    # TODO
    pass

# tf modules
def model(data):
    """Construct the CNN model.

    Args:
        data: batch of 28x28 images to evaluate.

    Returns:
        softmax of logits.
    """

    # input layer

    # conv1:
        # 3x3 conv layer, 32 filters, stride of 1
    kernel = tf.get_variable('kernel', shape=[3, 3, 1, 32], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(data, kernel, [1, 1, 1, 1], 'SAME')
    bias1 = tf.get_variable('bias1', shape=[32], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.bias_add(conv, bias1)

    # Relu
    conv1 = tf.nn.relu(conv1)

    # batch normalization
    batch_mean1, batch_var1 = tf.nn.moments(conv1, [0])
    batch1 = tf.nn.batch_normalization(conv1, batch_mean1, batch_var1, None, None, 1e-6)

    # 2x2 max pooling
    pool1 = tf.nn.max_pool(batch1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # flatten
    reshape = tf.reshape(pool1, [-1, pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])

    # fc layer with 784 output units
    fc_weights1 = tf.get_variable('fc_weights1', shape=[reshape.shape[1], 784], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    fc_bias1 = tf.get_variable('fc_bias1', shape=[784], dtype=tf.float64, initializer=tf.constant_initializer(0.1)) #0.1 init so it doesn't die in relu backprop
    # Relu
    fc1 = tf.nn.relu(tf.matmul(reshape, fc_weights1) + fc_bias1)

    # fc layer with 10 output units (1 for each class)
    fc_weights2 = tf.get_variable('fc_weights2', shape=[784, 10], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    fc_bias2 = tf.get_variable('fc_bias2', shape=[10], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
    # softmax
    #softmax = tf.nn.softmax(tf.matmul(fc1, fc_weights2) + fc_bias2)
    out = tf.matmul(fc1, fc_weights2) + fc_bias2

    return out, kernel, fc_weights1, fc_weights2

def train_nn(x, y, trainData, trainTarget, validData, validTarget, testData, testTarget, num_epochs=5, batch_size=32, learning_rate=1e-4, reg=0):

    pred, kernel, fc_weights1, fc_weights2 = model(x)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, pred))
    reg_loss = tf.nn.l2_loss(kernel) + tf.nn.l2_loss(fc_weights1) + tf.nn.l2_loss(fc_weights2)
    loss = tf.reduce_mean(loss + reg * reg_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    batches_per_epoch = int(trainData.shape[0])//batch_size

    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []
        for epoch in range(num_epochs):
            #trainData, trainTarget = shuffle(trainData, trainTarget)

            epoch_loss = 0
            epoch_accuracy = 0
            for batch in range(batches_per_epoch):
                batch_x = trainData[batch*batch_size:\
                                    min((batch+1)*batch_size, \
                                        trainData.shape[0])]
                batch_y = trainTarget[batch*batch_size:\
                                    min((batch+1)*batch_size, \
                                        trainTarget.shape[0])]

                _, l, acc = sess.run([optimizer, loss, accuracy], \
                                     feed_dict={x: batch_x,\
                                                y: batch_y})
                epoch_loss += l
                epoch_accuracy += acc # might not need this

            print("Epoch {}".format(epoch), ", Loss = ", l,", Training accuracy = ", acc)
            valid_l, valid_a = sess.run([loss, accuracy], \
                                             feed_dict={x: validData,\
                                                        y: validTarget})
            print("Validation loss = ", valid_l, ", Validation accuracy = ", valid_a)

            # save data for plotting
            train_loss.append(l)
            valid_loss.append(valid_l)
            train_acc.append(acc)
            valid_acc.append(valid_a)
        print("Final test accuracy is ", (accuracy.eval({x:testData, y:testTarget})))

        #plot graphs
        plt.plot(range(num_epochs), train_loss, 'b', label='Training Loss')
        plt.plot(range(num_epochs), valid_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()

        plt.plot(range(num_epochs), train_acc, 'b', label='Training Accuracy')
        plt.plot(range(num_epochs), valid_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    # load data into variables
    trainData, validData, testData, trainTarget, validTarget, testTarget \
            = loadData()

    trainTarget_OH, validTarget_OH, testTarget_OH = convertOneHot(trainTarget, validTarget, testTarget)

    #TODO: reshape data, batch it (maybe do this in train_nn)
    trainData = np.reshape(trainData, [-1, 28, 28, 1])
    validData = np.reshape(validData, [-1, 28, 28, 1])
    testData = np.reshape(testData, [-1, 28, 28, 1])

    # setup placeholders
    x = tf.placeholder(tf.float64, shape=[None, 28, 28, 1], name='data')
    y = tf.placeholder(tf.float64, shape=[None, 10], name='label')

    # train model
    train_nn(x, y, trainData, trainTarget_OH, validData, validTarget_OH, testData, testTarget_OH, num_epochs=50, batch_size=32, learning_rate=1e-4, reg=0.1)
