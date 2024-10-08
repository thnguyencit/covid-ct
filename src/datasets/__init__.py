import torchvision
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.utils import shuffle
from PIL import Image
from . import open_source, open_source_zenodo
from src import utils as ut
import os
import os
import numpy as np

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


# nthai 2007 : add para im_size=512

def get_dataset(dataset_dict ,split, datadir, exp_dict, im_size, dataset_size=None, augmentation=None, 
        preprocessing=None, isZenodoOpenSource = False,):
    if isZenodoOpenSource == True:
        dataset = open_source_zenodo.OpenSourceCovid19_2(split=split, datadir=datadir, exp_dict=exp_dict,im_size=im_size,augmentation=augmentation, preprocessing=preprocessing,)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_tgt_dict = dataset.img_tgt_dict[25:25+dataset_size[split]]
        return dataset
    else:
        
        dataset = open_source.OpenSource(split=split, datadir=datadir, exp_dict=exp_dict,im_size=im_size,augmentation=augmentation, preprocessing=preprocessing,)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_tgt_dict = dataset.img_tgt_dict[25:25+dataset_size[split]]
        return dataset


# old code without im_size
# def get_dataset(dataset_dict, split, datadir, exp_dict, dataset_size=None, augmentation=None, 
#         preprocessing=None):
#     dataset = open_source.OpenSource(split=split, datadir=datadir, exp_dict=exp_dict,augmentation=augmentation, preprocessing=preprocessing)
#     if dataset_size is not None and dataset_size[split] != 'all':
#         dataset.img_tgt_dict = dataset.img_tgt_dict[25:25+dataset_size[split]]

#     return dataset
