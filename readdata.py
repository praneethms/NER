# -*- coding: utf-8 -*-

import joblib
import os

def _read_data(input_file):
    """Reads a BIO data
    @param input_file: relative path of the input file 
    
    returns an array of tuples(word,label)
    """
    with open(input_file) as f:
	lines = []
	words = []
	lst = []
	for line in f:
	    contents = line.strip()
	    word = line.strip().split(' ')[0]
	    label = line.strip().split(' ')[-1]
            if contents.startswith("-DOCSTART-"):
	        words.append('')
	        continue
            if word == '.':
                lst.append((word, label))
	        lines.append(lst)
		words = []
		lst = []
		continue
	    if len(word) > 0:
		tp = (word, label)
		lst.append(tp)
    return lines


if __name__ == "__main__":
    print("Reading BERT data")
    filename = os.path.join("data","train.txt")
    contents = _read_data(filename)
	
    joblib.dump(contents, "data/bert-train.pkl")
    print("Conversion done..") 
