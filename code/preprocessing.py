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
#import nilearn.image as image
from nilearn import datasets

np.set_printoptions(threshold=sys.maxsize)

miyawaki_dataset = datasets.fetch_miyawaki2008()

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
    fmri_figure_filenames = miyawaki_dataset.func[:12]
    stimuli_random_runs_filenames = miyawaki_dataset.label[12:]
    stimuli_figures_filenames = miyawaki_dataset.label[:12]
    
    # shape of the binary (i.e. black and white values) image in pixels
    stimulus_shape = (10, 10)
    
    sys.stderr.write("Preprocessing data...")
    t0 = time.time()
    
    # Mask fMRI data: go from 4D to sample x features 
    masker = MultiNiftiMasker(mask_img=miyawaki_dataset.mask, detrend=True,
                              standardize=False)
    masker.fit()
    fmri_data = masker.transform(fmri_random_runs_filenames)
    fmri_figures_data = masker.transform(fmri_figure_filenames)
    
    print("The shape of the masked data is " + str(np.shape(fmri_data)))
    
    #Load the visual stimuli from csv files
    stimuli = []
    for y in stimuli_random_runs_filenames:
        stimuli.append(np.reshape(np.loadtxt(y, dtype=int, delimiter=','),
                                  (-1,) + stimulus_shape, order='F'))
    stimuli_figures = []
    for y in stimuli_figures_filenames:
        stimuli_figures.append(np.reshape(np.loadtxt(y, dtype=int, delimiter=','),
                             (-1,) + stimulus_shape, order='F'))

    # We now stack the fmri and stimulus data and remove an offset in the
    # beginning/end.
    fmri_data = np.vstack([x[2:] for x in fmri_data])
    stimuli = np.vstack([y[:-2] for y in stimuli]).astype(float)
    fmri_figures_data = np.vstack([x[2:] for x in fmri_figures_data])
    stimuli_figures = np.vstack([y[:-2] for y in stimuli_figures]).astype(float)
    
    # fmri_data is a matrix of *samples* x *voxels*
    print("Preprocessed fMRI data: " + str(fmri_data.shape[0]) + " samples x "+ str(fmri_data.shape[1])+" voxels")
    
    # Flatten the stimuli
    stimuli = np.reshape(stimuli, (-1, stimulus_shape[0] * stimulus_shape[1]))
    stimuli_figures = np.reshape(stimuli_figures, (-1, stimulus_shape[0] * stimulus_shape[1]))
    print("Preprocessed stimuli data: " + str(stimuli.shape[0]) + " samples x "+ str(stimuli.shape[1])+" pixels")
    
    #Figures
    print(str(stimuli_figures.shape[0]) + " geometrical figures")
    
    
    sys.stderr.write(" Done (%.2fs).\n" % (time.time() - t0))

    
    return fmri_data, stimuli, fmri_figures_data, stimuli_figures, masker