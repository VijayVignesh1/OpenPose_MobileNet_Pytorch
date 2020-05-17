import copy
import json
import math
import os
import pickle

import cv2
import numpy as np
import pycocotools

from torch.utils.data.dataset import Dataset


class TrainDataset(Dataset):
    def __init__(self, labels, stride, sigma, paf_thickness, transform=None):
        super().__init__()
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness
        self._transform = transform
        with open(labels, 'rb') as f:
            self._labels = pickle.load(f)

    def __getitem__(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(label['img_paths'], cv2.IMREAD_COLOR) 
        sample = {
            'label': label,
            'image': image,
        }
        if self._transform:
            sample = self._transform(sample)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        return sample

    def __len__(self):
        return len(self._labels)

    def _generate_keypoint_maps(self, sample):
        n_keypoints = 21
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        return keypoint_maps

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

class ValDataset(Dataset):
    def __init__(self, file_name, images_folder):
        super().__init__()
        # with open(labels, 'r') as f:
        #     self._labels = json.load(f)
        self._images_folder = images_folder
        self._images_folder = ""
        self.file_name=file_name


    def __getitem__(self, idx):
        # file_name = self._labels['images'][idx]['file_name']
        img = cv2.imread(os.path.join(self._images_folder, self.file_name), cv2.IMREAD_COLOR)
        # print("image",img.shape)
        return {
            'img': img,
            'file_name': self.file_name
        }

    def __len__(self):
        return 1
