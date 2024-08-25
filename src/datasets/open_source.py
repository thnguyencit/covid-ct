import torch
import os
import h5py
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom, tqdm
from . import transformers
from PIL import Image
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut

class OpenSource(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        datadir,
        exp_dict,
        im_size,
        augmentation=None, 
        preprocessing=None,
       
    ):
        # nthai 2007
        self.im_size = im_size

        self.exp_dict = exp_dict
        self.datadir = datadir
        self.split = split
        self.n_classes = 5

        self.img_path = os.path.join(datadir, 'OpenSourceDCMs')
        self.lung_path = os.path.join(datadir, 'LungMasks')
        self.tgt_path = os.path.join(datadir, 'InfectionMasks')

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.img_tgt_dict = []
        for tgt_name in os.listdir(self.tgt_path):
            # print("tgt_name")
            # print(tgt_name)
            lung_name = os.path.join(self.lung_path, tgt_name)
            scan_id, slice_id = tgt_name.split('_')
            slice_id = str(int(slice_id.replace('z', '').replace('.png', ''))).zfill(4)
            img_name = [f for f in os.listdir(os.path.join(self.img_path, 
                                    'DCM'+scan_id)) if 's%s' % slice_id in f][0]
            img_name = os.path.join('DCM'+scan_id, img_name)
            
            self.img_tgt_dict += [{'img': img_name, 
                                   'tgt': tgt_name,
                                   'lung': lung_name}]

        # get label_meta
        fname = os.path.join('./tmp', 'labels_array.pkl')
        if not os.path.exists(fname):
            labels_array = np.zeros((len(self.img_tgt_dict), 3))
            for i, idict in enumerate(tqdm.tqdm(self.img_tgt_dict)):
                img_name, tgt_name = idict['img'], idict['tgt']
                mask = np.array(Image.open(os.path.join(self.tgt_path, tgt_name)))
                uniques = np.unique(mask)
                if 0 in uniques:
                    labels_array[i, 0] = 1 
                if 127 in uniques:
                    labels_array[i, 1] = 1 
                if 255 in uniques:
                    labels_array[i, 2] = 1 
            hu.save_pkl(fname, labels_array)

        labels_array = hu.load_pkl(fname)
        # print(labels_array.shape)
        # self.np.where(labels_array[:,1:].max(axis=1))
        ind_list = np.where(labels_array[:,1:].max(axis=1))[0]
        self.img_tgt_dict = np.array(self.img_tgt_dict)[ind_list]
        if split == 'train':
            self.img_tgt_dict = self.img_tgt_dict[:300]
        elif split == 'val':
            self.img_tgt_dict = self.img_tgt_dict[300:]

    # nthai2007
    def __getitem__(self, i):
        out = self.img_tgt_dict[i]
        img_name, tgt_name, lung_name = out['img'], out['tgt'], out['lung']

        # read image
        image = read_xray(os.path.join(self.img_path, img_name))
        image = Image.fromarray(image).resize((self.im_size, self.im_size))
        image = np.array(image)
        # plt.imshow(image, cmap=plt.cm.bone) 
        # plt.show()
        # read infection mask
        tgt_mask = np.array(Image.open(os.path.join(self.tgt_path, tgt_name)).resize((self.im_size, self.im_size)).transpose(Image.FLIP_LEFT_RIGHT).rotate(90))
        # plt.imshow(image, cmap=plt.cm.bone) 
        # plt.show()
        # read lung mask
        lung_mask = np.array(Image.open(os.path.join(lung_name)).resize((self.im_size, self.im_size)).transpose(Image.FLIP_LEFT_RIGHT))
        mask = np.zeros(lung_mask.shape)
        mask[lung_mask!= 0] = 1
        mask[tgt_mask!= 0] = 2

        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # plt.imshow(image, cmap=plt.cm.bone) 
        # plt.show()

        image, mask = transformers.apply_transform(self.split, image=image, label=mask, 
                                       transform_name=self.exp_dict['dataset']['transform'], 
                                       exp_dict=self.exp_dict)

        # plt.imshow(image.squeeze().numpy(), cmap=plt.cm.bone) 
        # plt.show()
        # # apply augmentations

        # # # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # return torch.from_numpy(image), mask
        # return image, torch.LongTensor(mask)
        return {'images': image, 
                'masks': torch.LongTensor(mask),
                'meta': {'index': i,
                        #  'slice_thickness':img_dcm.SliceThickness, 
                        #  'pixel_spacing':str(img_dcm.PixelSpacing), 
                         'img_name': img_name, 
                         'tgt_name':tgt_name,
                         'image_id': i,
                         'split': self.split}}

    def __len__(self):
        return len(self.img_tgt_dict)



def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data