import numpy as np
import matplotlib.pyplot as plt
import dataset

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = dataset.get_data(N, D, K)
num_examples = N*K

# initialize parameters randomly
h = 100 # size of hidden layer

W = 0.01*np.random.randn(D,h)
b = np.zeros((1,h))

W2 = 0.01*np.random.randn(h,K)
b2 = np.zeros((1,K))

# hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
epochs = 10000

# gradient descent loop
for i in range(epochs):
  # evaluate hidden layer scores
  hidden_layer = np.maximum(0, np.dot(X, W) + b)
  # evaluate class scores
  scores = np.dot(hidden_layer, W2) + b2

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # compute the loss
  correct_logprobs = -np.log(probs[range(num_examples), y])
  loss = (np.sum(correct_logprobs) / num_examples) + 0.5*reg*np.sum(W2*W2)

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples

  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores) + reg*W2
  db2 = np.sum(dscores, axis=0, keepdims=True)

  # backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # ReLU activation
  dhidden[hidden_layer <= 0] = 0

  # finally backprop into W and b
  dW = np.dot(X.T, dhidden) + reg*W
  db = np.sum(dhidden, axis= 0, keepdims=True)

  # perform a parameter update
  W2 += -step_size*dW2
  b2 += -step_size*b2
  W += -step_size*dW
  b += -step_size*db

  # evaluate this iteration accuracy
  predicted_class = np.argmax(scores, axis=1)
  accuracy = np.mean(predicted_class == y)

  if i % 10 == 0:
    print("iteration: {} loss: {} accuracy: {}".format(i, loss, accuracy))

# visualize the data
fig = plt.figure()
cmap = plt.cm.get_cmap("Spectral")

plt.subplot(2, 1, 1)
plt.title("N: {}, D: {}, K:{}, h: {}".format(N, D, K, h))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap, edgecolors='k')

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]
Z = np.dot(np.maximum(0, np.dot(Xmesh, W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.subplot(2, 1, 2)
plt.title("Accuracy: {}".format(accuracy))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
# fig.savefig('net.png')