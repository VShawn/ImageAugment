from abc import abstractmethod
from ast import Str
from math import floor
from multiprocessing.context import assert_spawning
import sys
import getopt
import os
import json
import re
import time
import random
from PIL import Image
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo
from ClassifyPreprocess_SingleImageAugmenter import SingleImageAugmenter
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset, DataLoader


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ClassifyTraining_Dataset(TorchDataset):
    def __init__(self, image_paths: list[str], label_values: list[int], resize_to: int = None, random_crop_rate: tuple[float, float] = (0, 0.05), read_image_as_rgb_function=SingleImageAugmenter.open_image_by_opencv_as_rgb):
        self._image_paths = image_paths
        self._label_values = label_values
        self._out_image_size = resize_to
        self._random_crop_rate = random_crop_rate
        self._read_image_as_rgb_function = read_image_as_rgb_function
        self.transforms = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        rbg = self._read_image_as_rgb_function(self._image_paths[index], self._out_image_size)
        pil_image = Image.fromarray(rbg)
        tensor = self.transforms(pil_image)
        return self._label_values[index], tensor

    def __len__(self):
        return len(self._image_paths)

    @staticmethod
    def get_train_validate_image_list(label_info_csv_path: str, validate_ratio=0.2) -> tuple[list, list, list, list]:
        '''
        从数据集中生成训练集和测试集的文件路径列表
        调用方法：train_image_paths, train_image_labels, validate_image_paths, validate_image_labels = get_train_validate_image_list(...
        label_info_csv_path: DatasetAnalyser 生成的数据集标签描述 csv 文件路径
        validate_ratio: 提取多少数据作为验证集, 0.2 表示随机抽 20% 作为验证集
        '''
        assert(validate_ratio > 0.1)
        da = DatasetAnalyser.ReadFromCsv(label_info_csv_path)
        train_image_paths = []
        train_image_labels = []
        eval_image_paths = []
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
            eval_image_paths.extend(p1)
            validate_image_labels.extend(l1)
            train_image_paths.extend(p2)
            train_image_labels.extend(l2)
            assert(len(p1) + len(p2) == label.image_count)
        assert(len(train_image_paths) == len(train_image_labels))
        assert(len(eval_image_paths) == len(validate_image_labels))
        assert(os.path.exists(train_image_paths[0]))
        assert(os.path.exists(eval_image_paths[0]))
        return train_image_paths, train_image_labels, eval_image_paths, validate_image_labels

    @staticmethod
    def get_train_validate_data_loader(label_info_csv_path: str, input_image_size: int, batch_size: int, num_workers=4, validate_ratio=0.2, shuffle: bool = True) -> tuple[DataLoader, DataLoader]:
        '''
        从数据集中生成训练集和测试集
        label_info_csv_path: DatasetAnalyser 生成的数据集标签描述 csv 文件路径
        validate_ratio: 提取多少数据作为验证集, 0.2 表示随机抽 20% 作为验证集
        '''
        train_image_paths, train_image_labels, validate_image_paths, validate_image_labels = ClassifyTraining_Dataset.get_train_validate_image_list(label_info_csv_path, validate_ratio)
        train_dataset = ClassifyTraining_Dataset(train_image_paths, train_image_labels, input_image_size)
        validate_dataset = ClassifyTraining_Dataset(validate_image_paths, validate_image_labels, input_image_size)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), DataLoader(validate_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == '__main__':
    # 测试生成训练集和测试集
    tdl, vdl = ClassifyTraining_Dataset.get_train_validate_data_loader('D:\\UritWorks\\AI\\image_preprocess\\Augmented\\label_info.csv', 257, 32)
    print("data size:", len(tdl.dataset))
    print("loader len:", len(tdl))
    for i, (labels, images) in enumerate(tdl):
        print(labels)
        print(images.shape)
        if i > 10:
            break
