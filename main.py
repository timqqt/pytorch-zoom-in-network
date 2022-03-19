'''
Created on 2022-03-12
@author: Fanjie Kong
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import time
import torch
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
from math import floor, sqrt
from skimage.io import imsave
from torch.nn import functional as F
from dataset.custom_dataset import CustomDataReader
from dataset.colon_cancer_dataset import GigaPixelPatchwiseReader

from models.attention_models import Attention, AttentionOnAttention
from models.feature_extractors import FeatureExtractor
from models.classifier import Classifier

torch.set_default_tensor_type('torch.cuda.FloatTensor')



def get_models(args):
    
    if args.models_set == 'base':
        attention = Attention()
        aoa = AttentionOnAttention()
        fe = FeatureExtractor()
        clf = Classifier(args.num_classes)

    elif args.models_set == 'ResNet':
        attention = Attention()
        aoa = AttentionOnAttention()
        fe = FeatureExtractor()
        clf = Classifier(args.num_classes)
        
    model = ZoomInNet(
         attention, aoa, fe, clf,
         batch_size=batch_size,
         frame_size= args.tile_size,
         patch_size= args.patch_size,
         stage_1_level = np.log2(10),
         stage_2_level = np.log2(5),
         original_image_level = 0,
         num_classes=args.num_classes,
         reg_strength = args.reg_strength,
         n_patches = args.n_patches,
         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return attention, aoa, fe, clf, model

def load_configuratiton(args):
    config = dict()
    config['model_dir'] = args.model_dir
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-5)
    config['scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_ratio)
    config['criterion'] = nn.CrossEntropyLoss()
    config['num_epochs'] = args.num_epochs
    config['eval_every_epochs'] = 1
    config['eval_every'] = len(TrainingDataReader) * eval_every_epochs
    config['save_path'] = args.save_path
    config['clip'] = args.clipnorm
    config['training_show_every'] = len(TrainingDataReader) * 0.1
    config['contrastive_learning'] = args.contrastive
    config['apply_con_epochs']  = args.apply_con_epochs
    return config
    
def get_custom_dataset(args):
                                           
    train_at, valid_at, test_at =  load_annotations(args.dataset)
    train_loader = CustomDataReader(dataset_dir, annotation_name=args.train_annotations,
                                                  batch_size=args.batch_size, train=True)
    valid_loader = CustomDataReader(dataset_dir, annotation_name='labels_histo_valid_fold_'+str(i+1)+'.csv',
                                                    batch_size=1, train=False)
    test_loader = CustomDataReader(dataset_dir, annotation_name='labels_histo_valid_fold_'+str(i+1)+'.csv',
                                                batch_size=1, train=False)
    return train_loader, valid_loader, test_loader

def main(argv):
    parser = argparse.ArgumentParser(
        description=("Train a model with attention sampling on the "
                     "artificial mnist dataset")
    )
    parser.add_argument(
        "dataset",
        help="The directory that contains the dataset ",
    )
    parser.add_argument(
        "output",
        help="An output directory"
    )
                                           
    parser.add_argument(
    "--TenCrossValidation", default=True, action='store_true')
                                                                              
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",
        help="Choose the optimizer for Q1"
    )
    parser.add_argument(
        "--models_set",
        choices=["base", "ResNet"],
        default="base",
        help="Choose the different architecture of the zoom-in modules"
    )
    parser.add_argument(
        "--mode",
        choices=["10CrossValidation", "Training", "Evaluation"],
        default="10CrossValidation",
        help="working mode of the program"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Set the optimizer's learning rate"
    )
                                           
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=30,
        help="Set the decay steps of learning rate scheduler"
    )
    parser.add_argument(
        "--lr_decay_ratio",
        type=float,
        default=0.9,
        help="Set the decay ratio of learning rate scheduler"
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=5.0,
        help=("Clip the gradient norm to avoid exploding gradients "
              "towards the end of convergence")
    )

    parser.add_argument(
        "--patch_size",
        type=lambda x: tuple(int(xi) for xi in x.split("x")),
        default="27x27",
        help="Choose the size of the patch to extract from the high resolution"
    )

    parser.add_argument(
        "--n_patches",
        type=int,
        default=50,
        help="How many patches to sample"
    )
    parser.add_argument(
        "--regularizer_strength",
        type=float,
        default=0.01,
        help="How strong should the regularization be for the attention"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Choose the batch size for SGD"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="How many epochs to train"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Scale for downsampling images"
    )
    parser.add_argument(
        "--contrastive_learning", default=False, action='store_true')

    parser.add_argument(
        "--mode",
        choices=['train', 'test'],
        default='train',
        help="train mode or test mode"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="the name of the testing model"
    )
    parser.add_argument(
        "--see_attention",
        type=lambda x: x == 'True',
        default='True',
        help="whether output a attention map"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Overlap for eliminating boundary effects"
    )
    parser.add_argument(
        "--reg_coe",
        type=float,
        default=0,
        help="whether regularize the coefficients for dynamic sampling"
    )
    parser.add_argument(
        "--valid_step",
        default=1,
        help="how often we do the validation "
    )
    parser.add_argument(
        "--load_model",
        default='best',
        help="load which model"
    )

    parser.add_argument(
        "--gpu",
        default=0
    )
    parser.add_argument(
        "--num_works",
        default=4
    )
    args = parser.parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if args.mode == "10CrossValidation":
        total_cross = 10
        ten_cross_acc_list = []
        for i in range(total_cross):
            train_loader = HistoPatchwiseReader(args.dataset, annotation_name='labels_histo_train_fold_'+str(i+1)+'.csv',
                                                  batch_size=args.batch_size, train=True)
            valid_loader = HistoPatchwiseReader(args.dataset, annotation_name='labels_histo_valid_fold_'+str(i+1)+'.csv',
                                                            batch_size=1, train=False)
            model = get_models(args)
            configs = load_configuratiton(args)
            train_loss, train_acc, valid_loss, valid_acc = train(model, train_loader, valid_loader, configs, args)
            ten_cross_acc_list.append(valid_acc)
        print("Final 10 Cross Validation Results: ")
        print(ten_cross_acc_list)
        print("Average Accuracy: ", np.mean(ten_cross_acc_list))
    elif args.mode == "Training":
        train_loader, valid_loader, test_loader = get_custom_dataset(args)
        train(model, train_loader, valid_loader, configs, args)
        eval(model, test_loader, configs, args)
    elif args.mode == "Evaluation":
        test_loader = HistoPatchwiseReader(args.dataset, annotation_name='labels_histo_valid_fold_'+str(i+1)+'.csv',
                                                            batch_size=1, train=False)
        eval(model, test_loader, configs, args)
        
if __name__ == "__main__":
    main(None)

