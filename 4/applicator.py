#!/usr/bin/env python2
from viola import extract_features
import numpy as np
import scipy
import scipy.misc
import sklearn
import sys
import os

model = sys.argv[-2]
fname = sys.argv[-1]

def apply_adaboost(W, feature):
    result = 0
    for i in xrange(len(W)):
        h = 1 if feature[int(W[i][1])] > W[i][2] else -1
        result += h * W[i][0]
    return 1 if result > 0 else 0

def read_file(fname):
    image = scipy.misc.imread(fname)
    return image[:,:,0]

features = extract_features(read_file(fname))
w = np.loadtxt(model)
print apply_adaboost(w, features)
