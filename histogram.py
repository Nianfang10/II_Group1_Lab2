# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:00:03 2021

@author: ruswang
"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

h5 = h5py.File("P:\pf\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5", 'r')
INPT_1 = h5["INPT_1"]
INPT_2 = h5["INPT_2"]
INPT_3 = h5["INPT_3"]
INPT_4 = h5["INPT_4"]

NIR_1 = h5["NIR_1"]
NIR_2 = h5["NIR_2"]
NIR_3 = h5["NIR_3"]
NIR_4 = h5["NIR_4"]

rgb1 = np.zeros(3)
for i in range(3):
    #input = INPT_2[:,:,:,i]
    rgb1[i] = INPT_1[:,:,:,i].flatten()
    
plt.hist(rgb1, bins = 1000,label = ['r','g','b'])
plt.legend(loc = 'upper right')
plt.show
    
  

'''
maxrgb1 = np.zeros(3)
for i in range(3):
    maxd = np.zeros(20)
    for j in range(20):
        d = INPT_1[j,:,:,i]
        m_index = np.argmax(d)
        df = d.flatten()
        maxd[j] = df[m_index]
    #input = INPT_1[:,:,:,i]
    maxrgb1[i] = max(maxd)
    
print(maxrgb1)#[20541,18955,17639]

maxrgb2 = np.zeros(3)
for i in range(3):
    input = INPT_2[:,:,:,i]
    maxrgb2[i] = max(max(max(input)))
    
maxrgb3 = np.zeros(3)
for i in range(3):
    input = INPT_3[:,:,:,i]
    maxrgb3[i] = max(max(max(input)))
    
maxrgb4 = np.zeros(3)
for i in range(3):
    input = INPT_4[:,:,:,i]
    maxrgb4[i] = max(max(max(input)))
    
maxnir1 = max(max(max(input)))'''