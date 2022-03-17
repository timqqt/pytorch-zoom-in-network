import torch.nn as nn
# Need kind of clean up my code
import torch
import os
import time
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
from math import floor, sqrt
from skimage.io import imsave
from torch.nn import functional as F
from torch.distributions import Multinomial
from utils import SamplingPatches, MultinomialRegularizer, NCAMPatchwiseReader
from layers import ExpectationWithoutReplacement, ExpectationWithReplacement
from sampling import _sample_without_replacement, _sample_with_replacement
from networks import Attention, AttentionOnAttention, FeatureExtractor, Classifier
from utils import GigaPixelPatchwiseReader, SamplingPatches, MultinomialRegularizer, MSELoss
import six
import torch
import pandas as pd
import torch.nn as nn
import seaborn as sns
from itertools import chain
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Linear, Identity
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter, OrderedDict
# Save and Load Functions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
np.random.seed(1)
torch.manual_seed(1)


def SamplingPatchesV2(location, frame_location, source_image, sample_space, low_img_level, high_img_level, patch_size):
    '''

    :param location: (x,y)
    :param source_image:
    :param sample_space: shape of attention, like [100, 100]
    :return: shape [bs, C, H, W]
    '''
    # sampled location transform horizon expansion
    if len(location.shape) == 2:
        location = location[0]
    row = np.floor(location // sample_space[1])
    col = location % sample_space[1]
    # location - axis switch！ and linear map back
    x = col * (np.around(2**low_img_level)) + frame_location[0] -int(np.around(2**high_img_level)* patch_size[1]//2)
    y = row * (np.around(2**low_img_level)) + frame_location[1] -int(np.around(2**high_img_level)* patch_size[0]//2)
    patch_list = []
    for idx in range(x.size):
        patch = source_image.extract_patches(x[idx], y[idx], level=high_img_level, size=patch_size, show_mode='channel_first')
        patch_list.append(patch)
    patch_list = torch.stack(patch_list)
    return patch_list


def read_image(nargs):
    img_path, transform, device = nargs
    image = imread(img_path)
    if transform:
        image = transform(image)
    else:
        image = torch.tensor(image, dtype=torch.float, device=device)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
    return image

def image_path_processor_base(image_paths, source_path = '', transform=None, device="cuda"):
    img_list = []
    for img_path in image_paths:
        image = imread(os.path.join(source_path, img_path))
        if transform:
            image = transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float, device=device)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        img_list.append(image)
    img = torch.stack(img_list)
    return img

def image_path_processor_parallizer(image_paths, source_path = '', transform=None, device="cuda", num_of_threads=8):
    image_paths = [(os.path.join(source_path, e_ip), transform, device) for e_ip in image_paths]
    pool = multiprocessing.Pool(num_of_threads)
    img = pool.map(read_image, image_paths)
    pool.close()
    pool.join()
    return img

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook