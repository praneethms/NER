# -*- coding:utf-8 -*-
import pycrfsuite
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report


import joblib
import os
import sys



from features import *
import modeling as modeler

if __name__ == "__main__":
    if len(sys.argv) > 1  and sys.argv[1] == 'train': 
        traindata = joblib.load(os.path.join("data", "bert-train.pkl"))
        print(len(traindata))
        modeler.traincrf( os.path.join("data","bert-train.pkl"), 
	      os.path.join("data","bert-test.pkl")) 
    else:
        testdata = joblib.load(os.path.join("data","bert-test.pkl"))
        print(len(testdata))
        y_test, y_pred, y_prob = modeler.predict(os.path.join("data","bert-test.pkl"))

        rep = modeler.report(y_test, y_pred) 
	print rep
