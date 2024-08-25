from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut
from torchsummary import summary

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy, shutil
cudnn.benchmark = True



def test(exp_dict, savedir_base, datadir,  num_workers=0, scan_id=None,savedir=''):
    # bookkeepting stuff
    # ==================
    model_path = os.path.join(savedir, 'model_best.pth')
    # Dataset
    # ==================
    # val set
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])
    if str(scan_id) != 'None':
        test_set.active_data = test_set.get_scan(scan_id)
    test_sampler = torch.utils.data.SequentialSampler(test_set)
    test_loader = DataLoader(test_set,
                            sampler=test_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    # Model
    # ==================
    # chk = torch.load('best_model.ckpt')
    if torch.cuda.is_available():
        model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=test_set).cuda()
    else:
        model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=test_set).cpu()
    epoch = -1


    if str(model_path) != 'None':
        model_path = model_path
        model.load_state_dict(hu.torch_load(model_path))
    else:
        try:
            exp_dict_train = copy.deepcopy(exp_dict)
            del exp_dict_train['test_mode']
            savedir_train = os.path.join(savedir_base, hu.hash_dict(exp_dict_train))
            model_path = os.path.join(savedir_train, "model_best.pth")
            score_list = hu.load_pkl(os.path.join(savedir_train, 'score_list_best.pkl'))
            epoch = score_list[-1]['epoch']
            print('Loaded model at epoch %d with score %.3f' % epoch)
            model.load_state_dict(hu.torch_load(model_path))
        except:
            pass

    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    with open(os.path.join(savedir, "params.txt"), "w") as f:
        f.write(str(pytorch_total_params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', default='open_source_pspnet',nargs="+")
    parser.add_argument('-sb', '--savedir_base', default='CovidSeg/save')
    parser.add_argument('-d', '--datadir', default='CovidSeg/dataset')
    parser.add_argument("-r", "--reset",  default=1, type=int)
    parser.add_argument("-ei", "--exp_id", default='unetplus_timm-resnest26d')
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-ec", "--encoder", default='') # timm-efficientnet-b0
    parser.add_argument("-si", "--scan_id", type=str, default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in [args.exp_group_list]:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

        # format them for test
        for exp_dict in exp_list:
            exp_dict['test_mode'] = 1

    # Run experiments or View them
    # ----------------------------
    for exp_dict in exp_list:
        # do trainval
        test(exp_dict=exp_dict,
            savedir_base=args.savedir_base,
            datadir=args.datadir,
            num_workers=args.num_workers,
            scan_id=args.scan_id,
            savedir=savedir)
