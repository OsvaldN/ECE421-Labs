import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True
k = 3
d = dim
iterations = 300

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]
    print('val size: ',str(len(val_data)))
    print('train size: ',str(len(data)))

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs:
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs:
    # pair_dist: is the pairwise distance matrix (NxK)
    X = tf.expand_dims(X, 0)
    MU = tf.expand_dims(MU, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(X, MU)), 2)
    return distances

def bump_membership(X, MU):
    # Find the nearest centroid for each sample
    distances = distanceFunc(X, MU)
    # extract nearest cluster
    membership = tf.argmin(distances, 0)
    return membership

def plotter(X, MU, membership):
    plt.scatter(X[:, 0], X[:, 1], c=membership, s=50, alpha=0.5)
    plt.plot(MU[:, 0], MU[:, 1], 'kx', markersize=15)
    #plt.show()

def loss_function(X, MU):
    return tf.reduce_sum(tf.square(tf.reduce_min(distanceFunc(X, MU), axis=0)))


#X = tf.constant(data, dtype=tf.float32)
X = tf.placeholder(tf.float32, [None, dim])
means = tf.get_variable('means',
                         initializer=tf.random.normal((k,d), mean=0.0, stddev=1.0, dtype = tf.float32))

cost = loss_function(X, means)
optimizer = tf.train.AdamOptimizer(1e-1, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(cost)

memberships = bump_membership(X, means)

loss_vals=[]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    #writer = tf.summary.FileWriter("graph", sess.graph)
    sess.run(init)
    #X_vals = sess.run(X)
    #sess.run(membership)
    for i in range(iterations):
        loss, _ = sess.run([cost, optimizer], feed_dict={X: data})
        loss_vals.append(loss)
        '''
        sess.run(update_means)
        membership_vals = sess.run(update_membership)
        loss_val = sess.run(loss)
        loss_vals.append(loss_val)
        '''
        if (i+1) % 100==0:
            print("iteration" + str(i))
    mean_vals = means.eval()
    membership_vals = sess.run(memberships, feed_dict={X: data})
    #mean_vals = sess.run(update_means)
    #writer.close()
plotter(data, mean_vals, membership_vals)

plt.figure()
plt.plot(loss_vals)
plt.show()

 #tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99,epsilon=1e-5)
