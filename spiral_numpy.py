# spiral.py
# numpy version of classifier
# 
import numpy as np
import matplotlib.pyplot as plt

def input_data():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j # one-hot
    return X, y

X, y = input_data()

#Train a Linear Classifier
D, h, K = 2, 100, 3
# initialize parameters randomly
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):

    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))
    # compute the gradient on scores
    d_scores = probs
    d_scores[range(num_examples),y] -= 1
    d_scores /= num_examples
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    d_W2 = np.dot(hidden_layer.T, d_scores)
    d_b2 = np.sum(d_scores, axis=0, keepdims=True)
    # next backprop into hidden layer
    d_hidden = np.dot(d_scores, W2.T)
    # backprop the ReLU non-linearity
    d_hidden[hidden_layer <= 0] = 0
    # finally into W,b
    d_W = np.dot(X.T, d_hidden)
    d_b = np.sum(d_hidden, axis=0, keepdims=True)
    # add regularization gradient contribution
    d_W2 += reg * W2
    d_W  += reg * W
    # perform a parameter update
    W  += -step_size * d_W
    b  += -step_size * d_b
    W2 += -step_size * d_W2
    b2 += -step_size * d_b2

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.5f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1).reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
