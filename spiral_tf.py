import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def input_data():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros((N*K,K), dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.5 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix,j] = 1 # one-hot
    return X, y

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, b, W2, b2):
    hidden_layer = tf.nn.relu(tf.matmul(X, W) + b)
    scores = tf.matmul(hidden_layer, W2) + b2
    return scores

x_data, y_data = input_data()

D, h, K = 2, 100, 3

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, D])
Y = tf.placeholder(tf.float32, shape=[None, K])

W  = init_weights([D,h])
b  = init_weights([1,h])
W2 = init_weights([h,K])
b2 = init_weights([1,K])

py_x = model(X, W, b, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer().minimize(cost)
predict_op = tf.argmax(py_x, axis=1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(20000):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})
        if i % 1000 == 0:
            cost_val, pred_val = sess.run([cost, predict_op],
                 feed_dict={X: x_data, Y: y_data})
            accuracy = np.mean(np.argmax(y_data, axis=1) == pred_val)
            print("iteration %5d : loss %.5f : accuracy %.5f" % (i, cost_val, accuracy))

    # plot the resulting classifier
    h = 0.02
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    xgrid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = sess.run(predict_op, feed_dict={X: xgrid_data})
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x_data[:, 0], x_data[:, 1])
    # , c=np.int(np.argmax(y_data)), s=40, cmap=plt.cm.Spectral)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
