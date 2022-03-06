#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 18:09:11 2022

@author: pierrehouzelstein
"""
import matplotlib.pyplot as plt
import nibabel
from nilearn import datasets
import numpy as np
from sklearn.preprocessing import Binarizer

miyawaki_dataset = datasets.fetch_miyawaki2008()

def plt_fmri_stim(X, Y, pathfile):
    # Load image
    bg_img = nibabel.load(miyawaki_dataset.background)
    
    bg = bg_img.get_fdata()
    X = X.get_fdata()
    Y = np.reshape(Y,(10,10))
    #Transparency
    alpha = Binarizer().fit_transform(np.absolute(X)[..., 10].T)
    #print(alpha)
    """
    plt.imshow(bg[..., 10].T, origin='lower',
               interpolation='nearest', cmap='gray')
    plt.imshow(X[..., 10].T, origin='lower',
               interpolation='nearest', cmap='hot', alpha=alpha)"""
    
    fig = plt.figure()
    sp1 = fig.add_subplot(1,2,1)
    sp2 = fig.add_subplot(1,2,2)
    sp1.imshow(bg[..., 10].T, origin='lower',
               interpolation='nearest', cmap='gray')
    sp1.imshow(X[..., 10].T, origin='lower',
               interpolation='nearest', cmap='hot', alpha=alpha)
    plt.axis('off')
    sp2.imshow(Y, cmap = plt.cm.gray, interpolation = 'nearest')
    plt.axis('off')
    plt.show()
    
    if pathfile == None:
        return
    else:
        fig.savefig(pathfile)
    
    return

import re

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]
 
    