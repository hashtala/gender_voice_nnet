# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:04:49 2019

@author: gela
"""

import pandas as pd
import numpy as np
from Nafo import basic_functions as bsf
from Nafo.artificial_neural_net import Artificial_Neural_Net as ann
import theano.tensor as T


data = pd.read_csv('voice.csv')

X_train = data.iloc[:3000,:-1]
Y_train = data.iloc[:3000,-1]
'''
X, Y = bsf.shuffle(X,Y, 3168)
'''
'''
Y_train = Y[3000:3168, -1]
X_train = X[3000:3168,:-1]
'''
Y_train_enc = []



for i in range(len(Y_train)):
    if Y_train[i] == 'male':
        Y_train_enc.append(1)
    else:
        Y_train_enc.append(0)
        
del Y_train
Y_train = np.array(Y_train_enc)
del Y_train_enc

X_train, Y_train = bsf.shuffle(X_train, Y_train, 3000)

X_test = data.iloc[3000:3168, :-1]

model = ann([(20, 100, T.nnet.relu),
             (100, 50, T.nnet.relu),
             (50, 2, T.nnet.relu),
             (1,1, T.nnet.softmax)
             ]) 
    
    
info = model.fit(X_train,
          Y_train,
          lr = 2e-6,
          mu = 0.99,
          beta = 1,
          batches = 250,
          print_period = 500,
          step = 1000,
          epoch = 40000)

