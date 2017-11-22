import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)# initialize the gradient as zero
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
                dW[:,j] += X[i,:]
                dW[:,y[i]] += -X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train + 2*reg*W

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


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  S = np.dot(X,W)
  Sub = np.zeros(S.shape)
  for i in range(num_train):
    Sub[i] = num_classes*[S[i,y[i]] - 1]
  S -= Sub
  S = np.maximum(S,0)
  loss = np.sum(S)/num_train + reg*np.sum(np.square(W)) - 1#减1因为把第y[i]列也算进去了
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1),依次取每一个训练集的正确值
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[range(num_train), list(y)] = 0
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
  #不用循环就一步步处理（覆盖之前的值）

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #好好研读
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
