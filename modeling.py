
import pycrfsuite
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report


import joblib
import os
from features import *

_MODELFILE_ = os.path.join("models", "bert-eng.crfsuite")
def traincrf(trainfile, testfile):
    traindata = joblib.load(trainfile)
    testdata = joblib.load(testfile)
    
    X_train = [sent2features(s) for s in traindata]
    y_train = [sent2labels(s) for s in traindata]

    X_test = [sent2features(s) for s in testdata]
    y_test = [sent2labels(s) for s in testdata]
    	
    build_model(X_train, y_train)


def predict(testfile):
    testdata = joblib.load(testfile)
    X_test = [sent2features(s) for s in testdata]
    y_test = [sent2labels(s) for s in testdata]

    tagger = pycrfsuite.Tagger()
    tagger.open(_MODELFILE_)
    y_pred=[]
    y_prob=[]

    for x in X_test:
        y_pred_ = tagger.tag(x)
        y_pred.append(tagger.tag(x))
        for idx,k in enumerate(y_pred_):
	    y_prob.append(tagger.marginal(k,idx))

    tagger.close()
    return y_test, y_pred,y_prob       

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

def report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_ = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_ = lb.transform(list(chain.from_iterable(y_pred)))

    tags = set(lb.classes_) - {'O'}
    tags = sorted( tags, key=lambda tag: tag.split('-', 1)[::-1])
    class_idx = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(y_true_, y_pred_, labels = [class_idx[c] for c in tags],
		target_names=tags,)


