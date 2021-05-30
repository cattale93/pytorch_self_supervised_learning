import os
import numpy as np
from torch.utils.data import Dataset
from Lib.Datasets.processing.utility import one_hot_2_label_value
import random


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file overload the Dataset class of Pytorch
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class EUSARDataset(Dataset):
    """
    Implementation of the torch Dataset class for EUSAR dataset
    The Dataset must be structured as follow:
    The naming convetion for patches is: XXXX_XXXXXXX_data.npy
    Where:
        - XXXX is the tile ID (from 0 to N)
        - XXXXXXX is the patch ID (from zero to M)
        - data can be label rgb or radar
    Dataset
    --/ radar
        --/ radar patches in np format
    --/ rgb
        --/ rgb patches in np format
    --/ label
        --/ label patches in np format
    """
    def __init__(self, global_path, b_label=False, b_rgb=False, sar_c=5, opt_c=5, randomized=False):
        """
        Init function
        :param global_path: path to the directory of data (which should be composed of: rgb, radar, label)
        :param b_rgb: use label?
        :param b_rgb: use rgb data?
        :param b_rgb: use rgb data?
        :param sar_c: number of channel of sar data
        :param opt_c: number of channel of optical data
        :param randomized:
        """
        # global path to dataset folder
        self.data_path = os.path.join(global_path)
        # The dataset folder should always contain folders named as follow
        self.radar = "radar"
        self.label = "label"
        self.rgb = "rgb"
        self.b_label = b_label
        self.b_rgb = b_rgb
        self.sar_c = sar_c
        self.opt_c = opt_c
        self.randomized = randomized
        # create a list of the file in radar folder
        self.radar_list = sorted(os.listdir(os.path.join(self.data_path, self.radar)))
        # randomly sample file names to be able to access data randomly
        self.rand_list = random.sample(list(range(0, len(self.radar_list))), k=len(self.radar_list))
        # label and rgb are generated only if data are requested
        if self.b_label:
            self.label_list = sorted(os.listdir(os.path.join(self.data_path, self.label)))
        if self.b_rgb:
            self.rgb_list = sorted(os.listdir(os.path.join(self.data_path, self.rgb)))

    def __len__(self):
        """
        If b_label b_rgb are True we want both transcode and train for segmentation so the length is radar_list
        If b_label is False and b_rgb is True we want only to transcode so the length is radar_list
        If b_label is True and b_rgb is False we want only to segment so the length is label list
        :return: number of samples in dataset
        """
        if self.b_label and not self.b_rgb:
            return len(self.label_list)
        else:
            return len(self.radar_list)

    def __getitem__(self, idx):
        """
        this function return a dictionary of the data requested
        :param idx: index
        :return: dictionary with {"radar": radar, "label": label, "rgb": rgb, "name": name}
        """
        if self.randomized:
            idx = self.rand_list[idx]

        radar_name = self.radar_list[idx]
        radar = np.load(os.path.join(self.data_path, self.radar, radar_name), allow_pickle=True)[:self.sar_c]
        if self.b_label and not self.b_rgb:
            label_name = self.label_list[idx]
            label = np.load(os.path.join(self.data_path, self.label, label_name), allow_pickle=True)
            label = one_hot_2_label_value(label)
            return {"radar": radar, "label": label, "name": label_name}
        elif self.b_rgb and not self.b_label:
            rgb_name = self.rgb_list[idx]
            rgb = np.load(os.path.join(self.data_path, self.rgb, rgb_name), allow_pickle=True)[:self.opt_c]
            return {"radar": radar, "rgb": rgb, "name": radar_name}
        elif self.b_label and self.b_rgb:
            label_name = self.label_list[idx]
            label = np.load(os.path.join(self.data_path, self.label, label_name), allow_pickle=True)
            label = one_hot_2_label_value(label)
            rgb_name = self.rgb_list[idx]
            rgb = np.load(os.path.join(self.data_path, self.rgb, rgb_name), allow_pickle=True)[:self.opt_c]
            return {"radar": radar, "rgb": rgb, "label": label, "name": radar_name}
