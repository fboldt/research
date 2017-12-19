# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 05:34:34 2017

@author: francisco
"""
import random
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt

class MicroarrayDatasets():
    def __init__(self,debug=False):
        self.dsname = 'Microarray'
        self.path = "datasets/bio/debug" if debug else "datasets/bio/"
        self.filesname = [f for f in listdir(self.path) if isfile(join(self.path, f))]
    def files(self):
        for filename in self.filesname:
            yield join(self.path, filename),filename
    def datasets(self):
        for file,filename in self.files():
            data = genfromtxt(file)
            X,y = data[:,:-1],data[:,-1]
            yield X,y,filename

from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

class Experiment():
    def __init__(self,datasets=MicroarrayDatasets(),resdir="results",debug=False):
        self.ds = datasets
        self.resdir = resdir
        self.debug = debug
    def run(self, basename, clf, n_splits=10, n_repeats=10, random_state=None):
        for X,y,filename in self.ds.datasets():
            print(basename+' '+filename)
            resfilename = join(self.resdir, self.ds.dsname+'_'+filename+'.res_'+basename)
            file = open(resfilename, "a") if not self.debug else open(resfilename, "w")
            performances = []
            rskf = RepeatedStratifiedKFold(n_splits,n_repeats,random_state)
            for fold, idxs in enumerate(rskf.split(X, y)):
                if(fold >= file.tell()/10):
                    tr,te = idxs
                    clf.fit(X[tr],y[tr])
                    y_pred = clf.predict(X[te])
                    f1s = f1_score(y[te], y_pred, average='macro')
                    file.write('{:.7f}'.format(f1s)+'\n')
            file.close()

exp = Experiment(MicroarrayDatasets(True),debug=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
clfs ={'lr': LogisticRegression()}#,'rf': RandomForestClassifier()}

for cn, clf in clfs.items():
    exp.run(cn, clf, random_state=42, n_splits=3, n_repeats=1)
