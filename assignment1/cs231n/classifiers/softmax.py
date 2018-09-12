import numpy as np
from random import shuffle

def softmax(f, axis=None):
  f -= np.max(f)
  return np.exp(f) / np.sum(np.exp(f))

def softmax_vectorized(f):
  f -= np.max(f, axis=1)[:, np.newaxis]
  return np.exp(f) / np.sum(np.exp(f), axis=1)[:, np.newaxis]

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  D, C = W.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  for i in range(N):
    # Loss.
    f_i = X[i].dot(W)
    p_i = softmax(f_i)
    L_i = -np.log(p_i[y[i]])
    loss += L_i
    # Gradient.
    for j in range(10):
      if j == y[i]:
        dW[:, j] -= X[i]
      dW[:, j] += X[i] * p_i[j]
  loss = loss/N + 0.5 * reg * np.sum(W*W)
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  
  # Loss.
  f = X.dot(W)
  p = softmax_vectorized(f)
  losses = -np.log(p[np.arange(N), y])
  loss = np.sum(losses)/N + 0.5*reg*np.sum(W*W)
  
  # Gradient.
  # if j == y[i]: dW[:, j] -= X[i]
  # - subtract each row of X into the column of its true class.
  # - result[i, j] should be subtraced by ith feature of all instances for which
  #   j is the correct class.
  one_hot = p*0
  one_hot[np.arange(N), y] = 1
  dW = -X.T.dot(one_hot)
  # dW[:, j] += X[i] * p_i[j]
  # - add each row of X into each column of dW, weighted by column class prob.
  # - result[i, j] should have sum of ith feature in all examples
  #   weighted by their class probabilities.
  dW += X.T.dot(p)
  # OR from cs231n lectures.
  dscores = p
  dscores[range(N),y] -= 1
  dW = np.dot(X.T, dscores)
  #db = np.sum(dscores, axis=0, keepdims=True)
  # Divide by N and add contribution from regularization.
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

