import torch 
import tqdm
import torch.nn as nn 
from matplotlib import pyplot as plt
# Need kind of clean up my code
import os
import time
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
from cv2 import imread
from math import floor, sqrt
from skimage.io import imsave
from itertools import zip_longest
from torch.nn import functional as F
from new_utils import SamplingPatchesV2
import torch.distributions as dist
from torch.distributions import Multinomial
from utils import MultinomialRegularizer, NCAMPatchwiseReader
from layers import ExpectationWithoutReplacement, ExpectationWithReplacement
from sampling import _sample_without_replacement, _sample_with_replacement
from networks import Attention, AttentionOnAttention, FeatureExtractor, Classifier


def attention_inference_weights(attention):
    attention = (attention - attention.min())/attention.max()
    attention[attention < torch.quantile(attention, 0.3, dim=0, keepdim=True, interpolation='lower')] = 0
    return attention

    
class ZoomInNet(nn.Module):
    '''
    Zoom-in Net in a NN NutShell
    '''
    def __init__(self, 
                 attention, aoa, fe, clf,
                 batch_size=32,
                 frame_size= (250, 250),
                 patch_size= (50, 50),
                 low_low_img_level = 2,
                 low_img_level = 1,
                 high_img_level = 0,
                 num_classes=10,
                 reg_strength = 1e-4,
                 n_patches = 15,
                 weights = None,
                 contrasitve_learning=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.batch_size = batch_size
        self.attention = attention
        self.aoa = aoa
        self.fe = fe
        self.clf = clf
        self.expected_accum = ExpectationWithReplacement.apply
        self.expected = ExpectationWithoutReplacement.apply
        self.low_low_img_level = low_low_img_level
        self.low_img_level = low_img_level  
        self.high_img_level = high_img_level
        self.frame_size= frame_size
        self.patch_size= patch_size
        self.device = device
        self.reg_strength = reg_strength
        self.desired_patches = n_patches
        self.contrasitve_learning = contrasitve_learning
        self.weights = weights
        
    def _sample_with_replacement(self, logits, n_samples=10):
        '''
        Helper function to sample with replacement
        '''
#         distribution = dist.categorical.Categorical(logits=logits)
#         return distribution.sample(sample_shape=torch.Size([n_samples]))
        if self.training:
            return Multinomial(n_samples, logits).sample()
        else:
            #torch.topk(logits, n_samples)
            logits = attention_inference_weights(logits)
            #return torch.floor(n_samples * logits)
            return Multinomial(n_samples, logits).sample()
        
        
    def _sample_without_replacement(self, logits, n_samples=10):
        '''
        Helper function to sample without replacement
        '''
        if self.training:
            z = -torch.log(-torch.log(torch.rand_like(logits)))
            return torch.topk(logits+z, k=n_samples)[1]
        else:
            return torch.topk(logits, k=n_samples)[1]
    
    def _compute_loc_array(self, DataReader):
        '''
        Helper function to sample without replacement
        '''
        size_img = DataReader.get_size()
        frame_size = self.frame_size 
        patch_size = self.patch_size
        x = np.arange(0, size_img[0], frame_size[0])
        y = np.arange(0, size_img[1], frame_size[1])
        xx, yy = np.meshgrid(x, y)
        low_img_shape = xx.shape        
        high_img_shape = size_img
        loc_list = [(xx.flatten()[i], yy.flatten()[i]) for i in range(len(xx.flatten()[:]))]
        locs_array = np.array(loc_list)
        return locs_array
    
    def _computing_aoa(self, locs_array, dataReader):
        ''' 
        Helper funcntion to compute the attention-on-attention
        '''
        aoa_list = []
        for idx, loc in enumerate(locs_array):
            ## time - memory trading
            x, y = loc
            sub_image = dataReader.get_patches(x, y, level=self.low_low_img_level,
                                                   size=np.asarray(self.frame_size) // 2 ** self.low_low_img_level,
                                                   show_mode='channel_first')
            sub_image = sub_image.to(self.device)
            sub_image_tensor = torch.Tensor(sub_image).to(self.device)
            sub_image_tensor = sub_image_tensor.unsqueeze(0)
            sub_aoa = self.aoa(sub_image_tensor)
            aoa_list.append(sub_aoa)
        aoa_list = torch.cat(aoa_list)
        aoa_list = F.softmax(aoa_list)
        return aoa_list
    
    def _computing_attention(self, loc, dataReader):
        x, y = loc
        sub_image = dataReader.get_patches(x, y, level=self.low_img_level,
                                               size=np.asarray(self.frame_size) // 2 ** self.low_img_level,
                                               show_mode='channel_first')
        sub_image = sub_image.to(self.device)
        sub_image_tensor = torch.Tensor(sub_image).to(self.device)
        sub_image_tensor = sub_image_tensor.unsqueeze(0)
        sub_att = self.attention(sub_image_tensor)
        return sub_att
    
    def _compute_constrastive_predictions(self, x_reader):
        accum_feature_list = []
#         if self.contrasitve_learning:
#             con_accum_feature_list = []
        for e_reader in x_reader:
            now_dataReader = e_reader
            ## compute locs_array
            #free of size
            locs_array = self._compute_loc_array(now_dataReader)
            aoa_list = self._computing_aoa(locs_array, now_dataReader)
            reg_aoa = MultinomialRegularizer(aoa_list, self.reg_strength)
            # Sampling again implementation
            target_samples = self._sample_with_replacement(1 - aoa_list, self.desired_patches)
            target_idx = (target_samples != 0).nonzero()
            num_of_region = target_idx
            target_samples = target_samples[target_idx]
            target_aoa = aoa_list[target_idx].squeeze(dim=1)
            target_locs_array = locs_array[target_idx.cpu().numpy().astype(np.int)].squeeze()
            ##############################
            if len(target_locs_array.shape) == 1:
                # add an axis
                target_locs_array = np.expand_dims(target_locs_array, axis=0)

            expected_sub_features_list = []
            for idx, loc in enumerate(target_locs_array):
                sub_att = self._computing_attention(loc, now_dataReader)
                reg_aoa += MultinomialRegularizer(sub_att.view(-1), self.reg_strength)
                #sub_att = 1 - sub_att
                sampled_location = self._sample_without_replacement(1 - sub_att.view(sub_att.shape[0], -1),
                                                      int(target_samples[idx].cpu().numpy()))
                sampled_attention = sub_att.view(-1)[sampled_location]
                sampled_patches = SamplingPatches(sampled_location.cpu().numpy(), loc,
                                                  now_dataReader, sub_att.shape[2:],
                                                  self.low_img_level, self.high_img_level, self.patch_size)
                
                sub_feature = self.fe(sampled_patches.to(self.device))
                if self.weights is None:
                    self.weights = torch.ones_like(sampled_attention) / sampled_attention.shape[1]
                expected_sub_feature = self.expected(self.weights, sampled_attention, sub_feature.unsqueeze(0))
                expected_sub_features_list.append(expected_sub_feature.unsqueeze(0))
                # Re-init weights para
                self.weights = None
            # Double expectations
            # Repeat aoa based on the target samples
            expected_sub_feature_list = torch.cat(expected_sub_features_list, dim=1)
            weights_accum_f = target_samples.squeeze(1).unsqueeze(0) / self.desired_patches
            accum_features = self.expected_accum(weights_accum_f, target_aoa.unsqueeze(0), expected_sub_feature_list)
            accum_feature_list.append(accum_features)
        accum_feature_list = torch.cat(accum_feature_list, dim=0)
        con_prediction, _ = self.clf(accum_feature_list.to(self.device))
        return con_prediction, reg_aoa
        
    def forward(self, x_reader):
        '''
        x_reader : a list of datareader which has a length equal to batch size 
        
        return 
            - Predictions
        '''
        accum_feature_list = []
        for e_reader in x_reader:
            now_dataReader = e_reader
            ## compute locs_array
            #free of size
            locs_array = self._compute_loc_array(now_dataReader)
            aoa_list = self._computing_aoa(locs_array, now_dataReader)
            reg_aoa = MultinomialRegularizer(aoa_list, self.reg_strength)
            # Sampling again implementation
            
            target_samples = self._sample_with_replacement(aoa_list, self.desired_patches)
            #print(target_samples)
            target_idx = (target_samples != 0).nonzero()
            #print(target_idx)
            #print(locs_array.shape)
            num_of_region = target_idx
            target_samples = target_samples[target_idx]
            #print(target_samples)
            target_aoa = aoa_list[target_idx]
            #print(len(locs_array))
            #print(target_idx.cpu().data.numpy())
            target_locs_array = locs_array[target_idx.detach().cpu().numpy().astype(np.int)].squeeze()
            ##############################
            if len(target_locs_array.shape) == 1:
                # add an axis
                target_locs_array = np.expand_dims(target_locs_array, axis=0)

            expected_sub_features_list = []
            for idx, loc in enumerate(target_locs_array):
                sub_att = self._computing_attention(loc, now_dataReader)
                reg_aoa += MultinomialRegularizer(sub_att.view(-1), self.reg_strength)
                sampled_location = self._sample_without_replacement(sub_att.view(sub_att.shape[0], -1),
                                                      int(target_samples[idx].cpu().numpy()))
                sampled_attention = sub_att.view(-1)[sampled_location]
                sampled_patches = SamplingPatches(sampled_location.cpu().numpy(), loc,
                                                  now_dataReader, sub_att.shape[2:],
                                                  self.low_img_level, self.high_img_level, self.patch_size)
                
                sub_feature = self.fe(sampled_patches.to(self.device))
                if self.weights is None:
                    self.weights = torch.ones_like(sampled_attention) / sampled_attention.shape[1]
                expected_sub_feature = self.expected(self.weights, sampled_attention, sub_feature.unsqueeze(0))
                expected_sub_features_list.append(expected_sub_feature.unsqueeze(0))
                # Re-init weights para
                self.weights = None
            # Double expectations
            # Repeat aoa based on the target samples
            expected_sub_feature_list = torch.cat(expected_sub_features_list, dim=1)
            weights_accum_f = target_samples.squeeze(1).unsqueeze(0) / self.desired_patches
            accum_features = self.expected_accum(weights_accum_f, target_aoa.unsqueeze(0), expected_sub_feature_list)
            accum_feature_list.append(accum_features)
        accum_feature_list = torch.cat(accum_feature_list, dim=0)
        prediction, _ = self.clf(accum_feature_list.to(self.device))
        return prediction, reg_aoa