import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data100D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = False
k = 30
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
    sess.run(init)
    for i in range(iterations):
        loss, _ = sess.run([cost, optimizer], feed_dict={X: data})
        loss_vals.append(loss)
    mean_vals = means.eval()
    membership_vals = sess.run(memberships, feed_dict={X: data})
    if is_valid:
        valid_loss = sess.run(cost, feed_dict={X: val_data})

counts = np.bincount(membership_vals)
total = np.sum(counts)
fraction = counts/total

plt.scatter(data[:, 0], data[:, 1], c=membership_vals, s=25, alpha=0.75)
plt.plot(mean_vals[:, 0], mean_vals[:, 1], 'kx', markersize=10)
plt.title(str(k)+ '-Means Distribution Visualization')

plt.figure()
plt.title(str(k)+ '-Means Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('SSE Loss')
plt.plot(loss_vals)

if dim == 2:
    for i in range(len(fraction)):
        print(mean_vals[i])
        print(fraction[i])
elif dim == 100:
    print('k = ', k)
    print('Final Loss: ', loss)

if is_valid:
    print('k = ', k)
    print('validation loss: ', valid_loss)

plt.show()

