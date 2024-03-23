#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt

class Neural_network:

  def __init__(self, h0, h1, h2):
    self.h0 = h0
    self.h1 = h1
    self.h2 = h2

  def sigmoid(self, z):
    return np.divide(1, 1 + np.exp(-z))

  def d_sigmoid(self, z):
    return self.sigmoid(z) * (1-self.sigmoid(z))

  def loss(self, y_pred, Y):
    return  -np.sum(Y * np.log(y_pred) + (1-Y) * np.log(1-y_pred)) / (Y.shape[1])

  def init_params(self):
    W1 = np.random.normal(0,np.sqrt(np.divide(2, (self.h0 + self.h1))), size=(self.h1, self.h0))
    W2 = np.random.normal(0,np.sqrt(np.divide(2, (self.h1 + self.h2))), size=(self.h2, self.h1))
    b1 = np.random.normal(0,np.sqrt(np.divide(2, (self.h0 + self.h1))), size=(self.h1, 1))
    b2 = np.random.normal(0, np.sqrt(np.divide(2, (self.h1 + self.h2))), size=(self.h2, 1))
    return W1, W2, b1, b2

  def forward_pass(self, X, W1, W2, b1, b2):
    Z1 = W1.dot(X) + b1
    A1 = self.sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = self.sigmoid(Z2)
    return A2, Z2, A1, Z1

  def backward_pass(self, X, Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
    dW2 = (-Y+A2) @ A1.T
    dW1 = (W2.T * (-Y+A2)  * self.d_sigmoid(A1)) @ X.T
    db2 = np.sum(-Y+A2, axis=1, keepdims=True)
    db1 = (A1 * (1-A1)) @ (A2-Y).T * W2.T
    return dW1, dW2, db1, db2

  def accuracy(self, y_pred, y):
    y_pred = (y_pred>=0.5).astype(int)
    acc = np.mean(y_pred == y)
    return acc


  def predict(self, X, W1, W2, b1, b2):
    A2, _, _, _ = self.forward_pass(X, W1, W2, b1, b2)
    return A2

  def update(self, X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2, dW1, dW2, db1, db2, alpha ):
    grad_W1, grad_W2, grad_b1, grad_b2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
    W1 = W1 - alpha * grad_W1
    W2 = W2 - alpha * grad_W2
    b1 = b1 - alpha * grad_b1
    b2 = b2 - alpha * grad_b2
    return W1, W2, b1, b2

  def plot_decision_boundary(self, W1, W2, b1, b2):
    x = np.linspace(-0.5, 2.5,100 )
    y = np.linspace(-0.5, 2.5,100 )
    xv , yv = np.meshgrid(x,y)
    xv.shape , yv.shape
    X_ = np.stack([xv,yv],axis = 0)
    X_ = X_.reshape(2,-1)
    A2, Z2, A1, Z1 = self.forward_pass(X_, W1, W2, b1, b2)
    plt.figure()
    plt.scatter(X_[0,:], X_[1,:], c= A2)
    plt.show()

  def fit(self, X_train, Y_train, X_test, Y_test):
    alpha = 0.001
    W1, W2, b1, b2 = self.init_params()
    n_epochs = 10000
    train_loss = []
    test_loss = []
    for i in range(n_epochs):
      ## forward pass
      A2, Z2, A1, Z1 = self.forward_pass(X_train, W1, W2, b1, b2)
      
      ## backward pass
      dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
      ## update parameters
      W1, W2, b1, b2 = self.update(X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2, dW1, dW2, db1, db2, alpha)

      ## save the train loss
      train_loss.append(self.loss(A2, Y_train))
      ## compute test loss
      A2, Z2, A1, Z1 = self.forward_pass(X_test, W1, W2, b1, b2)
      test_loss.append(self.loss(A2, Y_test))

    ## plot boundary
      if i %1000 == 0:
        self.plot_decision_boundary(W1, W2, b1, b2)

    ## plot train et test losses
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()

    y_pred = self.predict(X_train, W1, W2, b1, b2)
    train_accuracy = self.accuracy(y_pred, Y_train)
    print ("train accuracy :", train_accuracy)

    y_pred = self.predict(X_test, W1, W2, b1, b2)
    test_accuracy = self.accuracy(y_pred, Y_test)
    print ("test accuracy :", test_accuracy)

