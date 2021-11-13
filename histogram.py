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

r1 = np.zeros(10980*10980*20)
g1 = np.zeros(10980*10980*20)
b1 = np.zeros(10980*10980*20)
nir1 = np.zeros(10980*10980*20)
for i in range(20):
    #input = INPT_2[:,:,:,i]
    #r1[i*10980*10980:(i+1)*10980*10980]= INPT_1[i,:,:,0].flatten()
    #g1[i*10980*10980:(i+1)*10980*10980]= INPT_1[i,:,:,1].flatten()
    #b1[i*10980*10980:(i+1)*10980*10980]= INPT_1[i,:,:,2].flatten()
    nir1[i*10980*10980:(i+1)*10980*10980]= NIR_1[i,:,:].flatten()
    
    
#plt.hist(rgb1, bins = 1000,label = ['r','g','b'])
plt.hist(r1,bins = 1000,alpha = 0.3,label = 'r')
plt.hist(g1,bins = 1000,alpha = 0.3,label = 'g')
plt.hist(b1,bins = 1000,alpha = 0.3,label = 'b')
plt.hist(nir1,bins = 1000,alpha = 0.3,label = 'nir')
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