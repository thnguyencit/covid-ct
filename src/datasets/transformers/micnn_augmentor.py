# obtained from https://github.com/frankkramer-lab/MIScnn

import numpy as np

from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms import ContrastAugmentationTransform, GaussianNoiseTransform
from batchgenerators.transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms import CutOffOutliersTransform, SimulateLowResolutionTransform


class Data_Augmentation:
    # Configurations for the data augmentation techniques
    config_p_per_sample = 0.15                      # Probability a data augmentation technique
                                                    # will be performed on the sample
    config_mirror_axes = (0, 1, 2)
    config_contrast_range = (0.3, 3.0)
    config_brightness_range = (0.5, 2)
    config_gamma_range = (0.7, 1.5)
    config_gaussian_noise_range = (0.0, 0.05)
    config_elastic_deform_alpha = (0.0, 900.0)
    config_elastic_deform_sigma = (9.0, 13.0)
    config_rotations_angleX = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    config_rotations_angleY = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    config_rotations_angleZ = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
    config_scaling_range = (0.85, 1.25)
    # Cropping settings
    cropping = False
    cropping_patch_shape = None
    # Segmentation map augmentation
    seg_augmentation = True

    def __init__(self, cycles=1, scaling=True, rotations=True,
                 elastic_deform=False, mirror=True, brightness=True,
                 contrast=True, gamma=True, gaussian_noise=True, cutoff=True, simres=True):
        # Parse parameters
        self.cycles = cycles
        self.scaling = scaling
        self.rotations = rotations
        self.elastic_deform = elastic_deform
        self.mirror = mirror
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.gaussian_noise = gaussian_noise
        self.cutoff = cutoff
        self.simres = simres

    def run(self, img_data, seg_data):
        # Define label for segmentation for segmentation augmentation
        if self.seg_augmentation : 
            seg_label = "seg"
        else : 
            seg_label = "class"
        # Create a parser for the batchgenerators module
        data_generator = DataParser(img_data, seg_data, seg_label)
        # Initialize empty transform list
        transforms = []
        
        # Add Cut Off Outliers augmentation
        if self.cutoff:
            aug_cutoff = CutOffOutliersTransform(percentile_lower=10, percentile_upper=90, per_channel=True)
            transforms.append(aug_cutoff)
        # Add Cut Off Outliers augmentation
        if self.simres:
            aug_simres = SimulateLowResolutionTransform(per_channel=True)
            transforms.append(aug_simres)
        # Add mirror augmentation
        if self.mirror:
            aug_mirror = MirrorTransform(axes=self.config_mirror_axes)
            transforms.append(aug_mirror)
        # Add contrast augmentation
        if self.contrast:
            aug_contrast = ContrastAugmentationTransform(
                                        self.config_contrast_range,
                                        preserve_range=True,
                                        per_channel=True,
                                        p_per_sample=self.config_p_per_sample)
            transforms.append(aug_contrast)
        # Add brightness augmentation
        if self.brightness:
            aug_brightness = BrightnessMultiplicativeTransform(
                                        self.config_brightness_range,
                                        per_channel=True,
                                        p_per_sample=self.config_p_per_sample)
            transforms.append(aug_brightness)
        # Add gamma augmentation
        if self.gamma:
            aug_gamma = GammaTransform(self.config_gamma_range,
                                       invert_image=False,
                                       per_channel=True,
                                       retain_stats=True,
                                       p_per_sample=self.config_p_per_sample)
            transforms.append(aug_gamma)
        # Add gaussian noise augmentation
        if self.gaussian_noise:
            aug_gaussian_noise = GaussianNoiseTransform(
                                        self.config_gaussian_noise_range,
                                        p_per_sample=self.config_p_per_sample)
            transforms.append(aug_gaussian_noise)
        # Add spatial transformations as augmentation
        # (rotation, scaling, elastic deformation)
        if self.rotations or self.scaling or self.elastic_deform or \
            self.cropping:
            # Identify patch shape (full image or cropping)
            if self.cropping : patch_shape = self.cropping_patch_shape
            else : patch_shape = img_data[0].shape[0:-1]
            # Assembling the spatial transformation
            aug_spatial_transform = SpatialTransform(
                                    patch_shape,
                                    [i // 2 for i in patch_shape],
                                    do_elastic_deform=self.elastic_deform,
                                    alpha=self.config_elastic_deform_alpha,
                                    sigma=self.config_elastic_deform_sigma,
                                    do_rotation=self.rotations,
                                    angle_x=self.config_rotations_angleX,
                                    angle_y=self.config_rotations_angleY,
                                    angle_z=self.config_rotations_angleZ,
                                    do_scale=self.scaling,
                                    scale=self.config_scaling_range,
                                    border_mode_data='constant',
                                    border_cval_data=0,
                                    border_mode_seg='constant',
                                    border_cval_seg=0,
                                    order_data=3, order_seg=0,
                                    p_el_per_sample=self.config_p_per_sample,
                                    p_rot_per_sample=self.config_p_per_sample,
                                    p_scale_per_sample=self.config_p_per_sample,
                                    random_crop=self.cropping)
            # Append spatial transformation to transformation list
            transforms.append(aug_spatial_transform)
        # Compose the batchgenerators transforms
        all_transforms = Compose(transforms)
        # Assemble transforms into a augmentation generator
        augmentation_generator = SingleThreadedAugmenter(data_generator,
                                                         all_transforms)
        # Perform the data augmentation x times (x = cycles)
        aug_img_data = None
        aug_seg_data = None
        for i in range(0, self.cycles):
            # Run the computation process for the data augmentations
            augmentation = next(augmentation_generator)
            # Access augmentated data from the batchgenerators data structure
            if aug_img_data is None and aug_seg_data is None:
                aug_img_data = augmentation["data"]
                aug_seg_data = augmentation[seg_label]
            # Concatenate the new data augmentated data with the cached data
            else:
                aug_img_data = np.concatenate((augmentation["data"],
                                              aug_img_data), axis=0)
                aug_seg_data = np.concatenate((augmentation[seg_label],
                                              aug_seg_data), axis=0)
        # Transform data from channel-first back to channel-last structure
        # Data structure channel-first 3D:  (batch, channel, x, y, z)
        # Data structure channel-last 3D:   (batch, x, y, z, channel)
        aug_img_data = np.moveaxis(aug_img_data, 1, -1)
        aug_seg_data = np.moveaxis(aug_seg_data, 1, -1)
        # Return augmentated image and segmentation data
        return aug_img_data, aug_seg_data


class DataParser:
    # Initialization
    def __init__(self, img_data, seg_data, seg_label):
        # Transform data from channel-last to channel-first structure
        # Data structure channel-last 3D:   (batch, x, y, z, channel)
        # Data structure channel-first 3D:  (batch, channel, x, y, z)
        self.img_data = np.moveaxis(img_data, -1, 1)
        self.seg_data = np.moveaxis(seg_data, -1, 1)
        # Cache segmentation label
        self.seg_label = seg_label
        # Define starting thread id
        self.thread_id = 0
    # Iterator
    def __iter__(self):
        return self
    # Next functionality: Return the img and seg in batchgenerators format
    def __next__(self):
        bg_dict = {'data':self.img_data.astype(np.float32),
                   self.seg_label:self.seg_data.astype(np.float32)}
        return bg_dict
    # Batchgenerators thread functionality for multi-threading
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id