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
import nibabel
from nilearn.maskers import MultiNiftiMasker
from nilearn.masking import unmask
import matplotlib.pyplot as plt
#import nilearn.image as image
from nilearn import datasets

np.set_printoptions(threshold=sys.maxsize)

miyawaki_dataset = datasets.fetch_miyawaki2008()

def plt_background():
    # Load image
    bg_img = nibabel.load(miyawaki_dataset.background)
    bg = bg_img.get_fdata()
    # Keep values over 6000 as artificial activation map
    #act = bg.copy()
    #act[act < 6000] = 0.
    # Display the background
    plt.imshow(bg[..., 10].T, origin='lower',
               interpolation='nearest', cmap='gray')
    # Cosmetics: disable axis
    plt.axis('off')
    plt.show()
    return

def load_and_mask_miyawaki_data():
    """
    

    Returns
    -------
    fmri_data : numpy.ndarray
        Preprocessed fmri data: samples x voxels activation value.
    stimuli : numpy.ndarray
        Flattened 10x10 binary stimuli 
    masker : nilearn.maskers.multi_nifti_masker.MultiNiftiMasker
        Object used to delimit the area of interest we are working on in the brain.
        Will be needed to get images back from the data we produce

    """

    
    # training data starts after the first 12 files
    fmri_random_runs_filenames = miyawaki_dataset.func[12:]
    stimuli_random_runs_filenames = miyawaki_dataset.label[12:]
    
    # shape of the binary (i.e. black and white values) image in pixels
    stimulus_shape = (10, 10)
    
    sys.stderr.write("Preprocessing data...")
    t0 = time.time()
    
    # Mask fMRI data: go from 4D to sample x features 
    masker = MultiNiftiMasker(mask_img=miyawaki_dataset.mask, detrend=True,
                              standardize=False)
    masker.fit()
    fmri_data = masker.transform(fmri_random_runs_filenames)
    
    #Load the visual stimuli from csv files
    stimuli = []
    for y in stimuli_random_runs_filenames:
        stimuli.append(np.reshape(np.loadtxt(y, dtype=int, delimiter=','),
                                  (-1,) + stimulus_shape, order='F'))

    # We now stack the fmri and stimulus data and remove an offset in the
    # beginning/end.
    fmri_data = np.vstack([x[2:] for x in fmri_data])
    stimuli = np.vstack([y[:-2] for y in stimuli]).astype(float)
    
    # fmri_data is a matrix of *samples* x *voxels*
    print("Preprocessed fMRI data: " + str(fmri_data.shape[0]) + " samples x "+ str(fmri_data.shape[1])+" voxels")
    
    # Flatten the stimuli
    stimuli = np.reshape(stimuli, (-1, stimulus_shape[0] * stimulus_shape[1]))
    print("Preprocessed stimuli data: " + str(stimuli.shape[0]) + " samples x "+ str(stimuli.shape[1])+" pixels")
    
    sys.stderr.write(" Done (%.2fs).\n" % (time.time() - t0))
    
    return fmri_data, stimuli, masker

def inverse_transform_fmri_data(X):
    mask_img=miyawaki_dataset.mask
    new_img = unmask(X, mask_img, order="F")
    return new_img