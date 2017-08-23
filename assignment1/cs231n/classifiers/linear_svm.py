import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. (so C "classifiers" working at once)
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
    scores = X[i].dot(W) # dot the classifiers, output a vector in R^10 (class scores)
    correct_class_score = scores[y[i]] # check the correct score

    count_margin_broken = 0 # how many classes violated the margin?

    dW_i = np.zeros(dW.shape) # gradient for this particular training example

    for j in range(num_classes):
      if j == y[i]:
        continue # we're only checking the score margins of the C - 1 correct scores
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count_margin_broken += 1 # another classifier busted
        # now, that class gradient is equal to 1 * x_i
        dW_i[:,j] = X[i]

    # finally, the weights for the correct class gets shunted one times the data
    # for each other classifier that busted
    dW_i[:,y[i]] = -1 * count_margin_broken * X[i]

    # now we add this example to the summed gradient
    dW += dW_i

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # normalize (at the end you dumbass not every time in the loop)
  dW = dW / num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # And gradient.
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # SVM gradient makes sense here- scores are capped underneath by 0, and loss function
  # is difference between incorrect class score and correct class score (hinge loss)
  # meaning that if you want to cut loss, you need to make the incorrect class scores
  # smaller. How? By cutting their weights.


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

  # Compute scores first
  scores = X.dot(W) # should output a N x C matrix


  # size N vector of the scores of the correct classes
  correct_scores = scores[np.arange(X.shape[0]), y]

  # now let's subtract that (plus one) from everyone else. they should be less
  # than zero otherwise they're too big.
  mat = (scores.T - (correct_scores - 1)).T

  mat[np.arange(X.shape[0]), y] = 0 # don't count the correct term

  thresh = np.maximum(np.zeros_like(scores), mat) # knock out negative terms

  loss = np.sum(thresh)
  loss /= X.shape[0]

  # regularization
  loss += 0.5 * reg * np.sum(W * W)


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

  mask = (thresh > 0) * 1 # 1 if difference in scores > 0 put 1, 0 otherwise. size N x C

  # set correct score column of mask to be equal to the number of busted margins * -1
  mask[np.arange(X.shape[0]), y] = -1 * np.sum(mask, axis=1)


  # remember, for each example x_i vector, we find its score mask s_i, and then
  # do x_i outer s_i to yield a size D x C "weight gradient".

  # with the entire dataset, where X is size N x D and score_mask is size N x C,
  # this is equivalent to multiplying X.T and score_mask.

  dW = X.T.dot(mask) / X.shape[0]

  # reg

  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
