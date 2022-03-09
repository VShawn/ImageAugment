from ast import Str
from math import floor
from multiprocessing.context import assert_spawning
import sys
import getopt
import os
import json
import re
import time
import logging
import random
import cv2
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset, DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap


class ImageLoader:
    def __init__(self, expected_size, pre_image_size=0, random_crop_range: int = 0):
        self.pre_image_size = pre_image_size
        self.expected_size = expected_size
        self.transforms = transforms.ToTensor()
        if random_crop_range > 0:
            self.__call_proc = Proc_random_crop(pre_image_size, expected_size)
        else:
            self.__call_proc = Proc(expected_size)

    def __call__(self, path):
        image = self.__call_proc(path)
        return self.transforms(image)

    @staticmethod
    def OpenByOpenCvToRgb(path: str):
        bgr = cv2.imread(path)
        # BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb


class ClassifyTraning_Dataset(TorchDataset):
    def __init__(self, image_paths: list[str], label_values: list[int], out_image_size: int, random_crop_pixel_range: 2):
        self._image_paths = image_paths
        self._label_values = label_values
        self._out_image_size = out_image_size
        self.loader = ImageLoader(image_size, pre_image_size)

    def __getitem__(self, index):
        label, path = self.data[index]
        tensor = self.loader(self.root_path + path)
        return label, tensor

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_train_validate_dataset(label_info_csv_path: str, image_size: int, validate_ratio=0.2) -> tuple[TorchDataset, TorchDataset]:
        '''
        从数据集中生成训练集和测试集
        label_info_csv_path: DatasetAnalyser 生成的数据集标签描述 csv 文件路径
        validate_ratio: 提取多少数据作为验证集, 0.2 表示随机抽 20% 作为验证集
        '''
        assert(validate_ratio > 0.1)
        da = DatasetAnalyser.ReadFromCsv(label_info_csv_path)
        train_image_paths = []
        train_image_labels = []
        validate_image_paths = []
        validate_image_labels = []
        for label in da.labels:
            # 随机打乱文件路径
            random.shuffle(label.image_paths)
            # 抽取测试集
            validate_count = floor(label.image_count * validate_ratio)
            labels = [label.label_value for path in label.image_paths]
            p1 = label.image_paths[0:validate_count]
            p2 = label.image_paths[validate_count:]
            l1 = labels[0:validate_count]
            l2 = labels[validate_count:]
            validate_image_labels.extend(p1)
            validate_image_paths.extend(l1)
            train_image_paths.extend(p2)
            train_image_labels.extend(l2)
            assert(len(p1) + len(p2) == label.image_count)
        assert(len(train_image_paths) == len(train_image_labels))
        assert(len(validate_image_paths) == len(validate_image_labels))
        pass


if __name__ == '__main__':
    # 测试生成训练集和测试集
    ClassifyTraningDataset.get_train_validate_dataset('D:\\UritWorks\\AI\\image_preprocess\\Augmented\\label_info.csv')
