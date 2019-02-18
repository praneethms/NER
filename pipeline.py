# -*- coding:utf-8 -*-
import pycrfsuite

import joblib
import os

from features import *


def traincrf(trainfile, testfile):
    traindata = joblib.load(trainfile)
    testdata = joblib.load(testfile)
    
    X_train = [sent2features(s) for s in traindata]
    y_train = [sent2labels(s) for s in traindata]

    X_test = [sent2features(s) for s in testdata]
    y_test = [sent2labels(s) for s in testdata]
    	
    build_model(X_train, y_train)


def build_model(X_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=True)
    for X,y in zip(X_train, y_train):
	trainer.append(X,y)
   
    trainer.set_params({
	'max_iterations':100,
	'feature.possible_transitions': True
    })
    print("Training.....")
    trainer.train(os.path.join("models","bert-eng.crfsuite"))
    print("Training complete")

if __name__ == "__main__":
    traindata = joblib.load(os.path.join("data", "bert-train.pkl"))
    print(len(traindata))
    traincrf( os.path.join("data","bert-train.pkl"), 
	      os.path.join("data","bert-test.pkl")) 
   
