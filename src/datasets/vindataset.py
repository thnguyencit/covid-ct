import torch
import os
import h5py
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom
# from . import transformers
from PIL import Image
import pandas as pd 
import pathlib
import typing
import cv2 
from tqdm import tqdm
from detectron2.structures import BoxMode
import pickle

from detectron2.engine import DefaultPredictor, DefaultTrainer, launch

class MyTrainer(DefaultTrainer):
    pass 

class VinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split='train',
        datadir='VINAI_Chest_Xray',
        exp_dict=None,
    ):
        self.exp_dict = exp_dict
        self.datadir = datadir
        self.split = split
        self.n_classes = 14

        self.img_path = os.path.join(datadir, 'train', 'train')
        self.csv_path = os.path.join(datadir, 'train_downsampled.csv')

        self.train_df = pd.read_csv(self.csv_path)
        
        self.img_tgt_dict = []
        for tgt_name in os.listdir(self.img_path):
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
        # self.np.where(labels_array[:,1:].max(axis=1))
        ind_list = np.where(labels_array[:,1:].max(axis=1))[0]
        self.img_tgt_dict = np.array(self.img_tgt_dict)[ind_list]
        if split == 'train':
            self.img_tgt_dict = self.img_tgt_dict[:300]
        elif split == 'val':
            self.img_tgt_dict = self.img_tgt_dict[300:]

    def __getitem__(self, i):
        out = self.img_tgt_dict[i]
        img_name, tgt_name, lung_name = out['img'], out['tgt'], out['lung']

        # read image
        img_dcm = pydicom.dcmread(os.path.join(self.img_path, img_name))
        image = img_dcm.pixel_array.astype('float')

        # read infection mask
        tgt_mask = np.array(Image.open(os.path.join(self.tgt_path, tgt_name)).transpose(Image.FLIP_LEFT_RIGHT).rotate(90))
        
        # read lung mask
        lung_mask = np.array(Image.open(os.path.join(lung_name)).transpose(Image.FLIP_LEFT_RIGHT))
        mask = np.zeros(lung_mask.shape)
        mask[lung_mask!= 0] = 1
        mask[tgt_mask!= 0] = 2
        

        image, mask = transformers.apply_transform(self.split, image=image, label=mask, 
                                       transform_name=self.exp_dict['dataset']['transform'], 
                                       exp_dict=self.exp_dict)
       
        return {'images': image, 
                'masks': torch.LongTensor(mask),
                'meta': {'index': i,
                         'slice_thickness':img_dcm.SliceThickness, 
                         'pixel_spacing':str(img_dcm.PixelSpacing), 
                         'img_name': img_name, 
                         'tgt_name':tgt_name,
                         'image_id': i,
                         'split': self.split}}

    def __len__(self):
        return len(self.img_tgt_dict)


def get_vinbigdata_dicts(imgdir, train_df, train_data_type='', use_cache=False, debug=True, target_indices=None):

    debug_str = f"_debug{int(debug)}"
    train_data_type_str = f"_{train_data_type}"
    cache_path = pathlib.Path(".") / f"dataset_dicts_cache{train_data_type_str}{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # train_meta = pd.read_csv(imgdir / "train_meta.csv")
        train_meta = train_df
        if debug:
            train_meta = train_meta.iloc[:500]  # For debug....

        # Load 1 image to get image size.
        image_id = train_meta.loc[0, "image_id"]
        image_path = os.path.join(imgdir, 'train', f"{image_id}.png")
        # image_path = str(imgdir / "train" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, train_meta_row in tqdm(train_meta.iterrows(), total=len(train_meta)):
            record = {}

            # image_id, height, width = train_meta_row.values
            arr = train_meta_row.values
            image_id = arr[0]
            height = arr[-2]
            width = arr[-1]
            # filename = str(imgdir / "train" / f"{image_id}.png")
            filename = os.path.join(imgdir, 'train', f"{image_id}.png")
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            objs = []
            for index2, row in train_df.query("image_id == @image_id").iterrows():
                # print(row)
                # print(row["class_name"])
                # class_name = row["class_name"]
                class_id = row["class_id"]
                if class_id == 14:
                    # It is "No finding"
                    # This annotator does not find anything, skip.
                    pass
                else:
                    # bbox_original = [int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])]
                    h_ratio = resized_height / height
                    w_ratio = resized_width / width
                    bbox_resized = [
                        int(row["x_min"]) * w_ratio,
                        int(row["y_min"]) * h_ratio,
                        int(row["x_max"]) * w_ratio,
                        int(row["y_max"]) * h_ratio,
                    ]
                    obj = {
                        "bbox": bbox_resized,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]
    return dataset_dicts

if __name__ == "__main__":
    # ds = VinDataset()
    imgdir = os.path.join(os.getcwd(), 'VINAI_Chest_Xray_1024')
    train_df = pd.read_csv(os.path.join(imgdir, 'train.csv'))
    get_vinbigdata_dicts(imgdir, train_df)