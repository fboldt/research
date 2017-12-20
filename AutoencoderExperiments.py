# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:23:47 2017

@author: francisco
"""

from keras.layers import Activation, Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.linear_model import LogisticRegression

class Autoencoder():
    def __init__(self,reduction=1):
        self.reduction = reduction
    def fit(self, X):
        m,n = X.shape
        encoding_dim = n//self.reduction if self.reduction>0 else n
        input_layer = Input(shape=(n,))
        #'''
        encoded = Dense(encoding_dim, activation='relu',
                        activity_regularizer=regularizers.l1(10e-4))(input_layer)
        '''
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        #'''
        decoded = Dense(n, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X,X)
    def predict(self, X):
        x = self.encoder.predict(X)
        return x
    def fit_transform(self,X):
        self.fit(X)
        return self.predict(X)

from sklearn.base import BaseEstimator
class ReductionModel(BaseEstimator):
    def __init__(self,levels=0,reduction=1):
        self.logreg = LogisticRegression()
        self.levels = levels
        self.reduction = reduction
    def fit(self, X, y):
        self.aes = []
        x = X
        m,n = x.shape
        for _ in range(self.levels):
            self.aes.append(Autoencoder(self.reduction))
            x = self.aes[-1].fit_transform(x)
        self.logreg.fit(x,y)
    def predict(self, X):
        x = X
        for ae in self.aes:
            x = ae.predict(x)
        y_pred = self.logreg.predict(x)
        return y_pred

'''
from numpy import genfromtxt
data = genfromtxt("datasets/bio/colon.m")
X,y = data[:,:-1],data[:,-1]

model = BasicModel()
model.fit(X,y)
y_pred = model.predict(X)
from sklearn.metrics import f1_score
f1s = f1_score(y, y_pred, average='macro')
print(f1s)
#'''