# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 18:55:04 2017

@author: francisco
"""

from numpy import genfromtxt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

filename = "datasets/bio/colon.m"
data = genfromtxt(filename)
X,y = data[:,:-1],data[:,-1]

class NeuralFeatRed():
    def __init__(self):
        self.lr = LogisticRegression()
    def fit(self,X,y):
        n_hidden,n_inputs = X.shape
        n_hidden = n_hidden//2
        n_outputs = n_inputs
        learning_rate = 0.01
        X_test = X_train = X
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        hidden = tf.layers.dense(X, n_hidden)
        outputs = tf.layers.dense(hidden, n_outputs)
        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(reconstruction_loss)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver() 
        n_iterations = 1000
        codings = hidden
        with tf.Session() as sess:
            init.run()
            for iteration in range(n_iterations):
                training_op.run(feed_dict={X: X_train})
            codings_val = codings.eval(feed_dict={X: X_train})
            self.saver.save(sess, "./nfr.ckpt")
        self.lr.fit(codings_val,y)
    def predict(self,X):
        n_hidden,n_inputs = X.shape
        X_test = X
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        codings = tf.layers.dense(X, n_hidden)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.saver.restore(sess, "./nfr.ckpt")
            codings_val = codings.eval(feed_dict={X: X_test})       
        y_pred = self.lr.fit(codings_val,y)
        return y_pred


class Model:
    def __init__(self, data, target):
        data_size = data.size #int(data.get_shape()[1])
        target_size = len(data) #int(target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(data.astype("float32"), weight) + bias
        self._prediction = tf.nn.softmax(incoming)
        cross_entropy = -tf.reduce_sum(target, tf.log(self._prediction))
        self._optimize = tf.train.RMSPropOptimizer(0.03).minimize(cross_entropy)
        mistakes = tf.not_equal(
            tf.argmax(target, 1), tf.argmax(self._prediction, 1))
        self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    @property
    def prediction(self):
        return self._prediction
    @property
    def optimize(self):
        return self._optimize
    @property
    def error(self):
        return self._error

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    model = Model(X,y)
    y_pred = model.predict(X)
f1s = f1_score(y, y_pred, average='macro')
