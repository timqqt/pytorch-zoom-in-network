from torchvision import transforms, datasets
from cv2 import imread, imwrite, resize, INTER_LINEAR
from skimage.io import imsave
from skimage.draw import polygon as ski_polygon
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu
from itertools import zip_longest
from math import floor
import openslide as ops
import warnings as ws
import numpy as np
import random
import torch
import h5py
import math
import cv2
import os
import time
import csv
import logging
import os
import xml.etree.ElementTree as Xml
from collections import OrderedDict, defaultdict, namedtuple
from typing import Sequence, Any, Tuple

import fnmatch
import logging
import os
from collections import namedtuple
from typing import Dict

from PIL import Image
from PIL import ImageDraw
from progress.bar import IncrementalBar

import openslide
from PIL import Image
class OpenImage:
    '''
    Read one histopathology image
    '''
    def __init__(self, directory, 
                 data_transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                            ])):
        self.img = np.asarray(imread(directory))
        self.data_transform = data_transform
        
    def read_region(self, pos, level, size):
        '''
        x, y are the cardinality axis: x for column and y for rows
        :param x: x location
        :param y: y location
        :param level: the view we are looking right now
        :param size: size of patch
        :return:
        '''
        x, y = pos
        factor = np.around(2 ** level)
        #print("in read_region", self.img.shape)
        patch = self.img[max(y, 0) : min(self.img.shape[0]-1, y+int(size[1]*factor)), 
                         max(x, 0) : min(self.img.shape[1]-1, x+int(size[0]*factor)), :]
        #print("in read_region patch 1", patch.shape)
        if patch.shape != size:
            patch = np.pad(patch, ((max(0-y, 0), max(min(y+int(size[1]*factor)+1-self.img.shape[0], int(size[1]*factor)), 0)), 
                                   (max(0-x, 0), max(min(x+int(size[0]*factor)+1-self.img.shape[1], int(size[0]*factor)), 0)), (0, 0)))
        if level != 0:
            patch = resize(patch, (int(patch.shape[0]//factor), int(patch.shape[1]//factor)), INTER_LINEAR)
        #print("in read_region", patch.shape)
        return patch
    
    def extract_patches(self, x, y, level=0, size=(50, 50), show_mode='channel_first'):
        '''
        Read patches from one gigapixel image
        Should read the patches from the center
        :return:
        '''
        '''
        return img.read_region((int(x), int(y)), level, size)
        '''

        this_patch = np.asarray(self.read_region((int(x), int(y)), level, size))[:, :, :3].astype(np.uint8)
        if self.data_transform is not None:
            this_patch = self.data_transform(this_patch)            
        return this_patch

        
    def get_mask(self, level=0):
        assert False, "get pixel-level annotationsn is not implemented"
        return  
    
    def get_patches(self, x, y, level=0, size=(0, 0), show_mode='channel_first'):
        '''
        :param x: a list of x-axis for patches. (batch_dim, [num_of_patches for one image])
        :param y: a list of y-axis for patches. (batch_dim, [num_of_patches for one image])
        :param level:
        :param size:
        :return:
        '''
        return self.extract_patches(x, y, level=level, size=size, show_mode=show_mode)
    
    def get_size(self):
        return self.img.shape

class OpenGigapixel:
    '''
    Read one Camelyon16 reader
    '''
    def __init__(self, directory, 
                 data_transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                            ])):
        self.img = ops.OpenSlide(os.path.join(directory))
        self.cur_name = directory
        self.data_transform = data_transform
    
    def extract_patches(self, x, y, level=0, size=(50, 50), show_mode='channel_first'):
        '''
        Read patches from one gigapixel image
        Should read the patches from the center
        :return:
        '''
        '''
        return img.read_region((int(x), int(y)), level, size)
        '''

        this_patch = np.asarray(self.img.read_region((int(x), int(y)), level, size))[:, :, :3].astype(np.uint8)
        if self.data_transform is not None:
            this_patch = self.data_transform(this_patch)            
        return this_patch

        
    def get_mask(self, level=0):
        assert False, "get pixel-level annotationsn is not implemented"
        return  
    
    def get_patches(self, x, y, level=0, size=(0, 0), show_mode='channel_first'):
        '''
        :param x: a list of x-axis for patches. (batch_dim, [num_of_patches for one image])
        :param y: a list of y-axis for patches. (batch_dim, [num_of_patches for one image])
        :param level:
        :param size:
        :return:
        '''
        return self.extract_patches(x, y, level=level, size=size, show_mode=show_mode)
    
    def get_size(self):
        return self.img.dimensions
    
    def _get_cur_name(self):
        return self.cur_name
    
    def find_roi_normal(self, rgb_image):
        # self.mask = cv2.cvtColor(self.mask, cv2.CV_32SC1)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([30, 30, 30])
        # [255, 255, 255]
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        #close_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        return image_open

    def get_preprocessed_locations_v2(self, frame_size, mask_level=4):
        '''
        :param frame_size: size of one frame
        :param mask_level:  the shrink level of the binary mask
        :return:
        '''
        # Binary the map
        # Get the location
        name = self._get_cur_name()
        binary_mask_dir = os.path.join(self._get_cur_name().replace('.tif', 'frame_size_'+str(frame_size[0])+'level_'+str(mask_level)+'_binary.png'))

        if os.path.isfile(binary_mask_dir):
            b_mask =  np.asarray(imread(binary_mask_dir))
        else:
            img = self.img
            lowimg = np.asarray(img.read_region((int(0), int(0)), mask_level, img.level_dimensions[mask_level]))
            mask = np.asarray(self.find_roi_normal(lowimg))
            imsave(binary_mask_dir, mask.astype(np.int), check_contrast=False)
            b_mask = mask
        size_img = self.get_size()
        x = np.arange(0, size_img[0], frame_size[0])
        y = np.arange(0, size_img[1], frame_size[1])
        xx, yy = np.meshgrid(x, y)
        loc_list = [(xx.flatten()[i], yy.flatten()[i]) for i in range(len(xx.flatten()[:]))]
        locs_array = np.array(loc_list)
        # filter
        filtered_locs_array = []
        for each_loc in locs_array:
            mapped_loc = np.floor(each_loc // 2**mask_level)
            ROI_exist = b_mask[int(mapped_loc[1]): int(mapped_loc[1])+int(np.floor(frame_size[0] // 2**mask_level)) - 1,
            int(mapped_loc[0]): int(mapped_loc[0]) + int(np.floor(frame_size[1] // 2**mask_level)) - 1].sum() > 0
            if ROI_exist:
                filtered_locs_array.append(each_loc)
        filtered_locs_array = np.array(filtered_locs_array)
        return filtered_locs_array
    
class CustomDataReader:
    '''
    
    '''
    # Now we only support batch size 1 just for simplicity
    def __init__(self, directory, annotation_name='annotations.csv', batch_size=1, train=False, reader_type=None):
        self.batch_size = batch_size
        self.directory = directory
        self.annotations = os.path.join(directory, annotation_name)
        self.cur_batch = None
        self.__cur_name = None
        self.train = train
        self.reader = OpenImage if reader_type is None else OpenGigapixel
        self.img_gen = self.generator()
        self.data_list = self._get_data()
        
        
    def __len__(self): return int(len(self.data_list)//self.batch_size)
    
    def _get_data(self):
        
        data_list = []
        annotations = np.loadtxt(self.annotations, delimiter='\n', dtype=str)
        for e_ann in annotations:
            img_dir, label = e_ann.split(',')
            data_list.append((img_dir, int(label)))
        return data_list

    def _batcher(self, inputs, batch_size=1, fillvalue=None):
        inputs = iter(inputs)
        args = [iter(inputs)] * batch_size
        return zip_longest(fillvalue=fillvalue, *args)

    def _get_cur_name(self):
        return self.__cur_name

    def generator(self):
        if self.train:
            np.random.shuffle(self.data_list)
        dataGenerator = self._batcher(self.data_list, self.batch_size)
        while dataGenerator:
            try:
                self.cur_batch = next(dataGenerator)
                self.__cur_name = self.cur_batch[0][0]
                self.cur_batch = self._DataPathtoData(self.cur_batch)
                yield self.cur_batch
            except StopIteration:
                print('Finished this epoch')
                break

    def _DataPathtoData(self, batch):
        '''
        :param batch:
        :return: reading data from the file path in batch
        '''
        new_batch = []
        #print(os.path.join(self.directory, batch[0][0]))
        for e_b in batch:
            if e_b is not None:
                new_batch.append((OpenHisto(os.path.join(self.directory, e_b[0])), e_b[1]))
        return new_batch