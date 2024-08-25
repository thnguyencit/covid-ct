import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'CovidSeg'))
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import pylab as plt
import exp_configs
import time
import numpy as np
import shutil

from src import models
from src import datasets
from src import utils as ut


import argparse
import albumentations as albu
from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True

def delete_and_backup_experiment(savedir, backup_flag=True):
    """Delete an experiment. If the backup_flag is true it moves the experiment
    to the delete folder.
    
    Parameters
    ----------
    savedir : str
        Directory of the experiment
    backup_flag : bool, optional
        If true, instead of deleted is moved to delete folder, by default False
    """
    # get experiment id
    exp_id = os.path.split(savedir)[-1]
    
    # get paths
    savedir_base = os.path.dirname(savedir)
    savedir = os.path.join(savedir_base, exp_id)

    if backup_flag:
        # create 'deleted' folder 
        dst = os.path.join(savedir_base, 'deleted', exp_id)
        os.makedirs(dst, exist_ok=True)

        if os.path.exists(dst):
            shutil.rmtree(dst)
    
    if os.path.exists(savedir):
        if backup_flag:
            # moves folder to 'deleted'
            shutil.move(savedir, dst)
        else:
            # delete experiment folder 
            shutil.rmtree(savedir)

    # make sure the experiment doesn't exist anymore
    assert(not os.path.exists(savedir))

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.3,
#         ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.3,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def pretrainval(pretrain_dir,exp_dict, savedir_base, datadir,im_size, reset=False, num_workers=0):
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    # exp_id = hu.hash_dict(exp_dict)
    exp_id = '{}_{}'.format(exp_dict['model']['base'], exp_dict['model']['encoder'])
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        try:
            hc.delete_and_backup_experiment(savedir)
        except:
            delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)


    # Dataset
    # ==================
    # train set
    # nthai 2007 : add para im_size
    
    if exp_dict['augmentation'] == False:
    
        train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                         split="train",
                                         datadir=datadir,
                                         exp_dict=exp_dict,
                                         im_size = im_size,
                                         dataset_size=exp_dict['dataset_size'],
                                         augmentation=None,
                                         isZenodoOpenSource = exp_dict['zenodo_ds'])
        # val set
        val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                       split="val",
                                       datadir=datadir,
                                       exp_dict=exp_dict,
                                       im_size = im_size,
                                       dataset_size=exp_dict['dataset_size'],
                                       augmentation=None,
                                       isZenodoOpenSource = exp_dict['zenodo_ds'])
        
    else :
        train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                         split="train",
                                         datadir=datadir,
                                         exp_dict=exp_dict,
                                         im_size = im_size,
                                         dataset_size=exp_dict['dataset_size'],
                                         augmentation=get_training_augmentation(),
                                         isZenodoOpenSource = exp_dict['zenodo_ds'])
        # val set
        val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                       split="val",
                                       datadir=datadir,
                                       exp_dict=exp_dict,
                                       im_size = im_size,
                                       dataset_size=exp_dict['dataset_size'],
                                       augmentation=get_validation_augmentation(),
                                       isZenodoOpenSource = exp_dict['zenodo_ds'])

    val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)
    # Model
    # ==================
    
    # nthai

    if torch.cuda.is_available():
        model = models.get_model(model_dict=exp_dict['model'],
                                exp_dict=exp_dict,
                                train_set=train_set).cuda()
    else:
        model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cpu()

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path_pretrain = os.path.join(pretrain_dir, "model.pth")
    score_list_path_pretrain = os.path.join(pretrain_dir, "score_list.pkl")
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path_pretrain):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path_pretrain))
        score_list = hu.load_pkl(score_list_path_pretrain)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))
    vis_loader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=num_workers,
                            collate_fn=ut.collate_fn)

    train_sampler = torch.utils.data.RandomSampler(
        train_set, replacement=True, num_samples=2*len(val_set))

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              collate_fn=ut.collate_fn,
                              batch_size=exp_dict["batch_size"], 
                              drop_last=True, num_workers=num_workers)

    s_time = time.time()

    for e in range(s_epoch, exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate and Visualize the model
        val_dict = model.val_on_loader(val_loader, 
                        savedir_images=os.path.join(savedir, "images"),
                        n_images=3, savedir=savedir)
        score_dict.update(val_dict)
        # model.vis_on_loader(
        #     vis_loader, savedir=os.path.join(savedir, "images"))

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = len(score_list)
        score_dict["time"] = time.time() - s_time

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        score_df.to_csv(os.path.join(savedir, 'score.csv'))

        # print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        if e == 0 or (score_dict.get("val_score", 0) > score_df["val_score"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.get_state_dict())
            print("Saved Best: %s" % savedir)

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', default='open_source_pspnet',nargs="+")
    parser.add_argument('-sb', '--savedir_base', default='CovidSeg/save')
    parser.add_argument('-d', '--datadir', default='CovidSeg/dataset')
    parser.add_argument("-r", "--reset",  default=1, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-ec", "--encoder", default='') # timm-efficientnet-b0
    parser.add_argument("-b", "--base", default='') # timm-efficientnet-b0 
    parser.add_argument("-w", "--weight", default='') # imagenet+5k 
    parser.add_argument("-bs", "--batch_size", type=int, default=2) # batch_size
    parser.add_argument("-t", "--test", type=bool, default=False)   
    parser.add_argument("-i", "--im_size", type=int, default=512) # image size for input
    parser.add_argument('-o', '--opt', default='adam') # optimizer adam or SGD 
    parser.add_argument('-ze', '--zenodo_ds',type=bool, default=False) # chose open source Dataset
    parser.add_argument('-ag', '--augmentation', type=bool, default=False) # augmentation
    parser.add_argument('-pd', '--pretrain_dir',  default='CovidSeg/save') # pretrain dir using for load training result
    parser.add_argument('-bi', '--binary',type=bool, default=False) # chose open source Dataset

    args = parser.parse_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in [args.exp_group_list]:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ===============
    
    for exp_dict in exp_list:
        # do trainval
        if args.encoder:
            exp_dict['model']['encoder'] = args.encoder
        if args.base:
            exp_dict['model']['base'] = args.base
        if args.weight:
            exp_dict['model']['weight'] = args.weight

        exp_dict['batch_size'] = args.batch_size
        exp_dict['test'] = args.test
        exp_dict["augmentation"] = args.augmentation
        exp_dict["zenodo_ds"] = args.zenodo_ds
        exp_dict["binary"] = args.binary

        pretrainval(pretrain_dir=args.pretrain_dir,
                exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                reset=args.reset,
                im_size = args.im_size,
                num_workers=args.num_workers)
