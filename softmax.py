import numpy as np
import matplotlib.pyplot as plt
import dataset

N = 108 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = dataset.get_data(N, D, K)

# hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# initialize parameters randomly
W = 0.01*np.random.randn(D,K)
b = np.zeros((1,K))

num_examples = N*K

for i in range(200):
  # evaluate class scores
  scores = np.dot(X, W) + b

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # Compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  # Compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # Backprop the gradient to the params (W,b) 
  dW = np.dot(X.T, dscores) + reg*W # regularization gradient
  db = np.sum(dscores, axis=0, keepdims=True)

  # Perform the param update
  W += -step_size * dW
  b += -step_size * db

  # Evaluate the training accuracy
  predicted_class = np.argmax(scores, axis=1)
  iteration_accuracy = np.mean(predicted_class == y)

  if i % 10 == 0:
    print("iteration: {} loss: {} accuracy: {}".format(i, loss, iteration_accuracy))


# lets visualize the data:
cmap = plt.cm.get_cmap("Spectral")
plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap)

# visualize decision boundary
h = 0.22
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]

scores = np.dot(Xmesh, W) + b
predicted_class = np.argmax(scores, axis=1)
Z = predicted_class.reshape(xx.shape)

plt.subplot(2, 1, 2)
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=cmap)

plt.show()
