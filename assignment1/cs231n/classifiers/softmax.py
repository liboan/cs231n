import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  # compute raw scores
  scores = X.dot(W) # scores should have shape (N, C)

  for i in range(N):
    f_i = scores[i,:] - np.max(scores[i,:])
    p = np.exp(f_i) / np.sum(np.exp(f_i)) # vector of probabilities

    # x_i's raw class scores stored in scores[i,:]
    loss += -1 * np.log(p[y[i]])

    for k in range(C):
      dW[:, k] += (p[k] - (k == y[i])) * X[i]


  # normalize
  loss /= N
  dW /= N

  # regularization
  loss += 0.5 * np.sum(W * W)
  dW += reg * W

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
  D = X.shape[1]
  C = W.shape[1]

  f = X.dot(W) # get scores, size (N, C)
  f -= np.max(f, axis=1, keepdims=True) # subtract out max of each datapt
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True) # because we're normalizing by
  # sums of all classes for each datapt, do that over all datapts
  p = np.exp(f) / sum_f # normalized probability vectors

  loss = np.sum(-np.log(p[np.arange(N), y])) # grab each one for y and sum up

  ind = np.zeros_like(p)
  ind[np.arange(N), y] = 1 # mask that highlights only the correct classes
  dW = X.T.dot(p - ind) # p or p-1 depending, per the formula

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
