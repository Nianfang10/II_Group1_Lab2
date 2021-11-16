from unicodedata import normalize

from numpy.lib.function_base import _interp_dispatcher
from torchvision.datasets.vision import VisionDataset
import h5py
import numpy as np
import os
import pdb
import torch
class SatelliteSet(VisionDataset):
    

    def __init__(self, windowsize=128, split='train'):
        self.wsize = int(windowsize)
        super().__init__(None)
        assert split in ['train','test','validate','small_train'], f'Split parameters "{split}" must be either "train" , "test", "validate" or "small_train". '
        if split == 'train' or 'validate':
            self.data_path = os.path.join('../data',f'dataset_rgb_nir_train.hdf5')
        else:
            self.data_path = os.path.join('../data',f'dataset_rgb_nir_{split}.hdf5')
        self.num_smpls, self.sh_x, self.sh_y = 3,10980,10980  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        # self.num_windows = 4 * self.sh_x / self.wsize * self.sh_y / self.wsize
        
        self.has_data = False
        self.split = split
        if split == 'train':
            self.num_windows = 3 * self.sh_x / self.wsize * self.sh_y / self.wsize
        else:
            self.num_windows = self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data(self,b):
        h5 = h5py.File(self.data_path, 'r')

        if b==0:
            self.CLD = h5["CLD_1"]
            self.INPT = h5["INPT_1"]
            self.NIR = h5["NIR_1"]
        elif b==1:
            self.CLD = h5["CLD_2"]
            self.INPT = h5["INPT_2"]
            self.NIR = h5["NIR_2"]
        elif b==2:
            self.CLD = h5["CLD_3"]
            self.INPT = h5["INPT_3"]
            self.NIR = h5["NIR_3"]
        elif b==3:
            self.CLD = h5["CLD_4"]
            self.INPT = h5["INPT_4"]
            self.NIR = h5["NIR_4"]
        
        self.GT = h5["GT"]
        
        
        self.has_data = True
        

    def __getitem__(self, index):

        b = index * 3 // self.num_windows
        if self.split == 'validate':
            b = 3
        elif self.split == 'test':
            b = 0
        if not self.has_data:
            self.load_data(b)
        
        """Returns a data sample from the dataset.
        """
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        # b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls
        
        # print(self.NIR[b][:, n:n + self.wsize, m:m + self.wsize])

        # crop all data at the previously determined position
       
        RGB_sample = self.INPT[:, n:n + self.wsize, m:m + self.wsize]
        NIR_sample = self.NIR[:, n:n + self.wsize, m:m + self.wsize]
        CLD_sample = self.CLD[:, n:n + self.wsize, m:m + self.wsize]

        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]
        
        CLD_Mask = (CLD_sample!=0)
        CLD_Mask = np.stack((CLD_Mask,CLD_Mask,CLD_Mask),axis = 3)
        RGB_sample = RGB_sample * CLD_Mask

        NZ1 = np.sum((RGB_sample!=0).astype(float)  ,0)
        NZ1[NZ1==0]=1
        #set 0 to 1 to make sure that the dominator is not 0
        RGB_sample = np.sum(RGB_sample,0)/NZ1

        NZ2 = np.sum((NIR_sample!=0).astype(float)  ,0)
        NZ2[NZ2==0]=1
        NIR_sample = np.sum(NIR_sample,0)/NZ2

        #normalize
        NIR_sample = np.asarray(NIR_sample, np.float16) / (2 ** 12 - 1)
        RGB_sample = np.asarray(RGB_sample, np.float16) / (2 ** 12 - 1)
        X_sample = np.concatenate([RGB_sample, np.expand_dims(NIR_sample, axis=-1)], axis=-1)

        #padding
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[-1])  # pad with -1 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))
        
        
        return np.asarray(x_sample),gt_sample

    def __len__(self):
        return self.num_windows
    
    def transform(self, x_sample, gt_sample):
        
        return np.asarray(x_sample),gt_sample


if __name__ == "__main__":
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    colormap = np.asarray(colormap)
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    



    dset = SatelliteSet( windowsize = 256, split='train')
    # create dataloader that samples batches from the dataset
    train_loader = torch.utils.data.DataLoader(dset,
                                               batch_size=8,
                                               num_workers=2,
                                               shuffle=False)

    # plot some examples
    f, axarr = plt.subplots(ncols=3, nrows=8)
    

    for x, y in tqdm(train_loader):
        x = np.transpose(x, [0, 2, 3, 1])
        y = np.where(y == 99, 3, y)
        for i in range(len(x)):
            axarr[i, 0].imshow(x[i, :, :, 1])
            axarr[i, 1].imshow(x[i, :, :, -1])
            axarr[i, 2].imshow(y[i])
        # plt.show()   
        
