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
  num_batch = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  for i in range(num_batch):
    scores[i,:] -= np.max(scores[i,:])
    loss += -np.log((np.exp(scores[i,y[i]]))/(sum(np.exp(scores[i,:]))))
    for j in range(num_classes):
        output = np.exp(scores[i,j])/sum(np.exp(scores[i,:]))
        if j == y[i]:
             dW[:,j] += (-1 + output) *X[i,:] 
        else: 
             dW[:,j] += X[i,:]*output 
        
  dW /= num_batch
  dW += reg*W
  loss /= num_batch
  loss += 0.5*reg*np.sum(np.square(W))
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
  num_batch = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  sum_n = np.log(np.sum(np.exp(shift_scores),axis = 1))#(N,)
  loss_matrix = -shift_scores[range(num_batch), list(y)] + sum_n
  loss = sum(loss_matrix)/num_batch + 0.5*reg*np.sum(np.square(W))
  
  coff = np.zeros((num_batch,num_classes))
  output = np.exp(shift_scores)/(np.sum(np.exp(shift_scores),axis = 1)).reshape(-1,1)
  coff[range(num_batch), list(y)] = -1
  coff += output
  dW = np.dot(X.T,coff)
  dW /= num_batch
  dW += reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

