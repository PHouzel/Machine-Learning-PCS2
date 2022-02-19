#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:25:31 2022

@author: pierrehouzelstein
"""
# Some basic imports
import time
import sys
import numpy as np
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets

def flatten(list_of_2d_array):
    flattened = []
    for array in list_of_2d_array:
        flattened.append(array.ravel())
    return flattened



def load_and_mask_miyawaki_data():
    """
    
    """
    # Load dataset
    miyawaki_dataset = datasets.fetch_miyawaki2008()

    sys.stderr.write("Fetching dataset...")
    t0 = time.time()
    
    # training data starts after the first 12 files
    fmri_random_runs_filenames = miyawaki_dataset.func[12:]
    fmri_figure_filenames = miyawaki_dataset.func[:12]
    stimuli_random_runs_filenames = miyawaki_dataset.label[12:]
    stimuli_figure_filenames = miyawaki_dataset.label[:12]
    y_shape = (10, 10)
    
    sys.stderr.write(" Done (%.2fs).\n" % (time.time() - t0))
    
    sys.stderr.write("Preprocessing data...")
    t0 = time.time()
    
    # Mask fMRI data: go from 4D to sample x features 
    masker = MultiNiftiMasker(mask_img=miyawaki_dataset.mask, detrend=True,
                              standardize=False)
    masker.fit()
    X_train = masker.transform(fmri_random_runs_filenames)
    X_test = masker.transform(fmri_figure_filenames)
    
    #Load the visual stimuli from csv files
    y_train = []
    for y in stimuli_random_runs_filenames:
        y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                                  (-1,) + y_shape, order='F'))

    y_test = []
    for y in stimuli_figure_filenames:
        y_test.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                                 (-1,) + y_shape, order='F'))

    X_train = np.vstack([x[2:] for x in X_train])
    y_train = np.vstack([y[:-2] for y in y_train]).astype(float)
    X_test = np.vstack([x[2:] for x in X_test])
    y_test = np.vstack([y[:-2] for y in y_test]).astype(float)

    n_pixels = y_train.shape[1]
    n_features = X_train.shape[1]
    
    # Build the design matrix for multiscale computation
    # Matrix is squared, y_rows == y_cols
    y_cols = y_shape[1]

    # Original data
    design_matrix = np.eye(100)


    # Example of matrix used for multiscale (sum pixels vertically)
    #
    # 0.5 *
    #
    # 1 1 0 0 0 0 0 0 0 0
    # 0 1 1 0 0 0 0 0 0 0
    # 0 0 1 1 0 0 0 0 0 0
    # 0 0 0 1 1 0 0 0 0 0
    # 0 0 0 0 1 1 0 0 0 0
    # 0 0 0 0 0 1 1 0 0 0
    # 0 0 0 0 0 0 1 1 0 0
    # 0 0 0 0 0 0 0 1 1 0
    # 0 0 0 0 0 0 0 0 1 1

    height_tf = (np.eye(y_cols) + np.eye(y_cols, k=1))[:y_cols - 1] * .5
    width_tf = height_tf.T

    yt_tall = [np.dot(height_tf, m) for m in y_train]
    yt_large = [np.dot(m, width_tf) for m in y_train]
    yt_big = [np.dot(height_tf, np.dot(m, width_tf)) for m in y_train]

    # Add it to the training set
    y_train = [np.r_[y.ravel(), t.ravel(), l.ravel(), b.ravel()]
               for y, t, l, b in zip(y_train, yt_tall, yt_large, yt_big)]

    y_test = np.asarray(flatten(y_test))
    y_train = np.asarray(y_train)

    # Remove rest period
    X_train = X_train[y_train[:, 0] != -1]
    y_train = y_train[y_train[:, 0] != -1]
    X_test = X_test[y_test[:, 0] != -1]
    y_test = y_test[y_test[:, 0] != -1]

    sys.stderr.write(" Done (%.2fs).\n" % (time.time() - t0))
    
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = load_and_mask_miyawaki_data()

print(X_train.shape())