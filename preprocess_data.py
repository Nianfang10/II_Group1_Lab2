# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:27:02 2021

@author: ruswang
"""

import numpy.ma as npm
#Remove cloud
from unicodedata import normalize
from torchvision.datasets.vision import VisionDataset
import h5py
import numpy as np
import os
import pdb
import torch
class SatelliteSet(VisionDataset):

    def __init__(self, windowsize=128, split='train'):
        self.wsize = windowsize
        super().__init__(None)
        assert split in ['train','test','validate','small_train'], f'Split parameters "{split}" must be either "train" , "test", "validate" or "small_train". '
        self.data_path = os.path.join('data',f'P:\pf\pfshare\data\mikhailu\dataset_rgb_nir_{split}.hdf5')#dataset_{split}_1.h5
        self.num_smpls, self.sh_x, self.sh_y = 3,10980,10980  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)
        self.has_data = False
        self.split = split

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data(self):
        h5 = h5py.File(self.data_path, 'r')
        if self.split == "test":
            #h5 = h5py.File(self.data_path, 'r')
            self.CLD_1 = h5["CLD_1"]
            self.CLD_2 = h5["CLD_2"]
            self.CLD_3 = h5["CLD_3"]
            self.CLD_4 = h5["CLD_4"]
            self.GT = h5["GT"]
            self.INPT_1 = h5["INPT_1"]
            self.INPT_2 = h5["INPT_2"]
            self.INPT_3 = h5["INPT_3"]
            self.INPT_4 = h5["INPT_4"]
            self.NIR_1 = h5["NIR_1"]
            self.NIR_2 = h5["NIR_2"]
            self.NIR_3 = h5["NIR_3"]
            self.NIR_4 = h5["NIR_4"]
            self.has_data = True
        if self.split == "validate":
            self.CLD_1 = h5["CLD_1"][:,7980:,7980:]
            self.CLD_2 = h5["CLD_2"][:,7980:,7980:]
            self.CLD_3 = h5["CLD_3"][:,7980:,7980:]
            self.CLD_4 = h5["CLD_4"][:,7980:,7980:]
            self.GT = h5["GT"][:,7980:,7980:]
            self.INPT_1 = h5["INPT_1"][:,7980:,7980:,:]
            self.INPT_2 = h5["INPT_2"][:,7980:,7980:,:]
            self.INPT_3 = h5["INPT_3"][:,7980:,7980:,:]
            self.INPT_4 = h5["INPT_4"][:,7980:,7980:,:]
            self.NIR_1 = h5["NIR_1"][:,7980:,7980:]
            self.NIR_2 = h5["NIR_2"][:,7980:,7980:]
            self.NIR_3 = h5["NIR_3"][:,7980:,7980:]
            self.NIR_4 = h5["NIR_4"][:,7980:,7980:]
            self.has_data = True
        if self.split == "train":
            CLD_1_c = np.copy(h5["CLD_1"])
            CLD_1_c[:,7980:,7980:] = 0
            self.CLD_1 = CLD_1_c
            CLD_2_c = np.copy(h5["CLD_2"])
            CLD_2_c[:,7980:,7980:] = 0
            self.CLD_2 = CLD_2_c
            CLD_3_c = np.copy(h5["CLD_3"])
            CLD_3_c[:,7980:,7980:] = 0
            self.CLD_3 = CLD_3_c
            CLD_4_c = np.copy(h5["CLD_4"])
            CLD_4_c[:,7980:,7980:] = 0
            self.CLD_4 = CLD_4_c
            GT_c = np.copy(h5["GT"])
            GT_c[:,7980:,7980:] = 0
            self.GT = GT_c
            INPT_1_c = np.copy(h5["INPT_1"])
            INPT_1_c[:,7980:,7980:,:] = 0
            self.INPT_1 = INPT_1_c
            INPT_2_c = np.copy(h5["INPT_2"])
            INPT_2_c[:,7980:,7980:,:] = 0
            self.INPT_2 = INPT_2_c
            INPT_3_c = np.copy(h5["INPT_3"])
            INPT_3_c[:,7980:,7980:,:] = 0
            self.INPT_3 = INPT_3_c
            INPT_4_c = np.copy(h5["INPT_4"])
            INPT_4_c[:,7980:,7980:,:] = 0
            self.INPT_4 = INPT_4_c
            NIR_1_c = np.copy(h5["NIR_1"])
            NIR_1_c[:,7980:,7980:] = 0
            self.NIR_1 = NIR_1_c
            NIR_2_c = np.copy(h5["NIR_2"])
            NIR_2_c[:,7980:,7980:] = 0
            self.NIR_2 = NIR_2_c
            NIR_3_c = np.copy(h5["NIR_3"])
            NIR_3_c[:,7980:,7980:] = 0
            self.NIR_3 = NIR_3_c
            NIR_4_c = np.copy(h5["NIR_4"])
            NIR_4_c[:,7980:,7980:] = 0
            self.NIR_4 = NIR_4_c
            

    def __getitem__(self, index):
        if not self.has_data:
            self.load_data()

        """Returns a data sample from the dataset.
        """
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x  # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls

        # crop all data at the previously determined position
        RGB_sample = self.RGB[b, n:n + self.wsize, m:m + self.wsize]
        NIR_sample = self.NIR[b, n:n + self.wsize, m:m + self.wsize]
        CLD_sample = self.CLD[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]

        # normalize NIR and RGB by maximumg possible value
        NIR_sample = np.asarray(NIR_sample, np.float32) / (2 ** 16 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)

        ### correct gt data ###
        # first assign gt at the positions of clouds
        cloud_positions = np.where(CLD_sample > 10)
        GT_sample[cloud_positions] = 2
        # second remove gt where no data is available - where the max of the input channel is zero
        idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
        GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training
        GT_sample = np.where(GT_sample > 3, 99, GT_sample)
        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[99])  # pad with 99 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        

        return torch.tensor(np.asarray(x_sample)), torch.tensor(gt_sample).long()

    def __len__(self):
        return self.num_windows

def normalize(sample):
    new_sample = (sample*1.0)/256.0
    return new_sample


if __name__ == "__main__":
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    colormap = np.asarray(colormap)
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    



    dset = SatelliteSet( windowsize = 256, split='train')
    # create dataloader that samples batches from the dataset
    train_loader = torch.utils.data.DataLoader(dset,
                                               batch_size=8,
                                               num_workers=8,
                                               shuffle=True)

    # Please note that random shuffling (shuffle=True) -> random access.
    # this is slower than sequential reading (shuffle=False)
    # If you want to speed up the read performance but keep the data shuffled, you can reshape the data to a fixed window size
    # e.g. (-1,4,128,128) and shuffle once along the first dimension. Then read the data sequentially.
    # another option is to read the data into the main memory h5 = h5py.File(root, 'r', driver="core")

    # plot some examples
    f, axarr = plt.subplots(ncols=3, nrows=8)
    

    for x, y in tqdm(train_loader):
        x = np.transpose(x, [0, 2, 3, 1])
        y = np.where(y == 99, 3, y)
        #print(x.shape)
        for i in range(len(x)):
            axarr[i, 0].imshow(x[i, :, :, :3])
            axarr[i, 1].imshow(x[i, :, :, -1])
            axarr[i, 2].imshow(colormap[y[i]] / 255)

        plt.show()
        #quit()


'''
def load_data():
    h5 = h5py.File("P:\pf\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5", 'r')
    CLD_1 = h5["CLD_1"]
    CLD_2 = h5["CLD_2"]
    CLD_3 = h5["CLD_3"]
    CLD_4 = h5["CLD_4"]
    GT = h5["GT"]
    INPT_1 = h5["INPT_1"]
    INPT_2 = h5["INPT_2"]
    INPT_3 = h5["INPT_3"]
    INPT_4 = h5["INPT_4"]
    NIR_1 = h5["NIR_1"]
    NIR_2 = h5["NIR_2"]
    NIR_3 = h5["NIR_3"]
    NIR_4 = h5["NIR_4"]
    has_data = True
    return NIR_1,CLD_1,INPT_1

NIR_1,CLD_1,INPT_1 = load_data()   
NIR_1m = npm.array(NIR_1[1], mask = CLD_1) 

if __name__ == "__main__":
    
    

class SatelliteSet(VisionDataset):

    def __init__(self, windowsize=128, test=False):
        self.wsize = windowsize
        super().__init__(None)
        self.num_smpls, self.sh_x, self.sh_y = 3,10980,10980  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)
        self.has_data = False

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data(self):
        h5 = h5py.File("P:\pf\pfshare\data\mikhailu\dataset_rgb_nir_train.hdf5", 'r')
        self.CLD_1 = h5["CLD_1"]
        self.CLD_2 = h5["CLD_2"]
        self.CLD_3 = h5["CLD_3"]
        self.CLD_4 = h5["CLD_4"]
        self.GT = h5["GT"]
        self.INPT_1 = h5["INPT_1"]
        self.INPT_2 = h5["INPT_2"]
        self.INPT_3 = h5["INPT_3"]
        self.INPT_4 = h5["INPT_4"]
        self.NIR_1 = h5["NIR_1"]
        self.NIR_2 = h5["NIR_2"]
        self.NIR_3 = h5["NIR_3"]
        self.NIR_4 = h5["NIR_4"]
        self.has_data = True

    def __getitem__(self, index):
        if not self.has_data:
            self.load_data()

        """Returns a data sample from the dataset.
        """
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x  # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls

        # crop all data at the previously determined position
        RGB_sample = self.RGB[b, n:n + self.wsize, m:m + self.wsize]
        NIR_sample = self.NIR[b, n:n + self.wsize, m:m + self.wsize]
        CLD_sample = self.CLD[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]

        # normalize NIR and RGB by maximumg possible value
        NIR_sample = np.asarray(NIR_sample, np.float32) / (2 ** 16 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)

        ### correct gt data ###
        # first assign gt at the positions of clouds
        cloud_positions = np.where(CLD_sample > 10)
        GT_sample[cloud_positions] = 2
        # second remove gt where no data is available - where the max of the input channel is zero
        idx = np.where(np.max(X_sample, axis=-1) == 0)  # points where no data is available
        GT_sample[idx] = 99  # 99 marks the absence of a label and it should be ignored during training
        GT_sample = np.where(GT_sample > 3, 99, GT_sample)
        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[99])  # pad with 99 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        return np.asarray(x_sample), gt_sample

    def __len__(self):
        return self.num_windows
'''