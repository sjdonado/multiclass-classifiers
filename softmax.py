import numpy as np
import matplotlib.pyplot as plt
import dataset

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = dataset.get_data(N, D, K)
num_examples = N*K

# initialize parameters randomly
W = 0.01*np.random.randn(D,K)
b = np.zeros((1,K))

# hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
epochs = 200

# gradient descent lopp
for i in range(epochs):
  # evaluate class scores
  scores = np.dot(X, W) + b

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples), y])
  data_loss = np.sum(correct_logprobs) / num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # backprop the gradient to the params (W,b) 
  dW = np.dot(X.T, dscores) + reg*W # regularization gradient
  db = np.sum(dscores, axis=0, keepdims=True)

  # perform the param update
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
plt.title("N: {}, D: {}, K:{}".format(N, D, K))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap, edgecolors='k')

# visualize decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]
scores = np.dot(Xmesh, W) + b
predicted_class = np.argmax(scores, axis=1)
Z = predicted_class.reshape(xx.shape)

plt.subplot(2, 1, 2)
plt.title("Accuracy: {}".format(accuracy))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
# fig.savefig('softmax.png')