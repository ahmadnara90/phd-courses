# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:43:18 2024

@author: masoo
"""
import numpy
from  matplotlib import pyplot as plt

def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def activation_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w-learning_rate*grad

x=0.1
target = 0.3
learning_rate = 0.1
w = numpy.random.rand()
print("Initial W: ", w)

old_err = 0
itr_num = []
preds = []
for k in range(80000):
    itr_num.append(k)
    #forward pass
    y = w * x
    predicted = sigmoid(y)
    err = error(predicted, target)
    preds.append(predicted)
    #backward pass
    g1 = error_predicted_deriv(predicted, target)
    g2 = activation_sop_deriv(predicted)
    g3 = sop_w_deriv(x)

    grad = g3*g2*g1
  #  print(predicted)

    w = update_w(w, grad, learning_rate)

plt.plot(itr_num,preds)
