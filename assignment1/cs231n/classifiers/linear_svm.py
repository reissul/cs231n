import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # TODO: update grad here.
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += reg*2.0*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  delta = 1.
  # scores \in NxC (500, 10)
  scores = X.dot(W)
  # truth \in N (500)
  truth = scores[np.arange(N), y]
  # margins \in NxC (500x10)
  margins = np.clip(scores.T - truth + delta, 0, None).T
  margins[np.arange(N), y] = 0
  loss = margins.sum()/N + reg*(W*W).sum()
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  counts = np.sum(margins > 0, axis=1) # counts \in N (500,)
  num_classes = X.shape[1]

  # Half-vectorized with iterations over classes.
  """
  dWj = dW*0
  dWy = dW*0
  for k in range(W.shape[1]):
    # wj = gradient contribution for each feature and label k
    #      where the kth classifier incorrectly fired.
    not_labeled_k_mask = (margins[:, k] > 0) # \in N
    not_labeled_k_instances = X[not_labeled_k_mask] # \in something x D
    wj = np.sum(not_labeled_k_instances, axis=0) # \in D
    # wy = gradient conntribution for each feature and true label
    #      where the kth classifier incorrectly fired.
    labeled_k_mask = (y == k) # \in N
    labeled_k_instances = X[labeled_k_mask, :] # \in something x D
    labeled_k_counts = counts[labeled_k_mask, np.newaxis] # \in something X 1
    wy = -np.sum(labeled_k_counts * labeled_k_instances, axis=0)
    dWj[:, k] = wj
    dWy[:, k] = wy
    dW[:, k] = wj + wy
  """

  # Vectorized.
  #   dW = X.T.dot(incorrect_mat) maps X[i] to dW[:,j] b times if
  #   incorrect_mat[i,j] = b.
  incorrect_mat = (margins > 0) * 1.
  incorrect_mat[np.arange(N), y] = -counts
  dW = X.T.dot(incorrect_mat)

  dW /= N
  dW += reg*2.0*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
