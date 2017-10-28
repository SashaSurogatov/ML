#!/usr/bin/env python2
import numpy as np
import scipy
import scipy.misc
import sys
import os
import math
from viola import extract_features

folder = sys.argv[-2]
model = sys.argv[-1]

def read_file(fname):
    image = scipy.misc.imread(fname)
    return image[:,:,0]

def decision_stump(x_train, y_train, D_train, d):
    F = sys.float_info.max
    theta = 0
    idx = 0
    for j in xrange(0, d):
        x = x_train[:,j]        
        sort_idx = x.argsort()
        x = x[sort_idx]
        y = y_train[sort_idx]
        D = D_train[sort_idx]
        
        indexes_1 = sort_idx[y[sort_idx]==1]
        F_t = np.sum(D[indexes_1])
        
        if F_t < F:
            F = F_t
            theta = x[0] - 1
            idx = j
        x = np.append(x, [x[-1] + 1])
        for i in xrange(len(x) - 1):
            F_t = F_t - y[i] * D[i]
            if F_t < F and x[i] != x[i + 1]:
                F = F_t
                theta = (x[i] + x[i + 1]) / 2.0
                idx = j
    return idx, theta

def adaboost(x, y, T):
    D = np.array([1.0/len(x)]*len(x))
    W = []
    for t in xrange(T):
        idx, theta = decision_stump(x, y, D, len(x[0]))
        h = []           
        err = 0
        for i in xrange(len(x)):
            h.append(1 if x[i][idx] > theta else -1)
            if h[i] != y[i]:
                err += D[i]
        weight = math.log(1.0 / err - 1) / 2.0
        Z = 2 * math.sqrt(err * (1 - err))
        for i in xrange(len(D)):
            D[i] = D[i] * math.exp(-weight * y[i] * h[i]) / Z
        W.append((weight, idx, theta))
    return W

x_train = []
y_train = []

for fname in os.listdir(os.path.join(folder, 'cars')):
    feature = extract_features(read_file(os.path.join(folder, 'cars', fname)))
    x_train.append(feature)
    y_train.append(-1)

for fname in os.listdir(os.path.join(folder, 'faces')):
    feature = extract_features(read_file(os.path.join(folder, 'faces', fname)))
    x_train.append(feature)
    y_train.append(1)

W = adaboost(np.array(x_train), np.array(y_train), 15)

with open(model, 'wt') as fout:
    np.savetxt(fout, np.array(W))