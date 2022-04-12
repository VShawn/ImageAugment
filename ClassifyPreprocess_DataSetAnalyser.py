import os
import json
from dataclasses import dataclass
import struct
from typing import List

from cv2 import RETR_CCOMP
from numpy import array

CONST_NeedTrain = "_NeedTrain"
CONST_BigPic = "_BigPic"


class LabelInfo(object):
    """
    标签信息，包括标签名称、标签值、图像数量、标签文件夹路径
    """

    def __init__(self, label_main_name: str, label_sub_index_name: str, label_value: int, label_dir_path: str):
        self.label_main_name = label_main_name
        self.label_sub_index_name = label_sub_index_name
        self.label_name = '{}{}'.format(label_main_name, label_sub_index_name)
        self.label_value = label_value
        self.label_dir_path = label_dir_path
        self.label_dir_name = os.path.basename(label_dir_path)
        self.image_paths = self.get_image_paths(label_dir_path)
        # 文件夹内的 bmp jpg 图片数量
        self.image_count = len(self.image_paths)

    @staticmethod
    def get_image_paths(dir_path: str) -> list[str]:
        '''
        返回指定目录下所有图片的路径
        '''
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            return [os.path.join(dir_path, file) for file in os.listdir(dir_path) if (file.lower().endswith('.bmp') or file.lower().endswith('.jpg') or file.lower().endswith('.png'))]
        return []


class DatasetAnalyser(object):
    """
    分析指定数据集文件夹下的文件夹名称，生成一个标签配置文件
    """

    def __init__(self):
        self.labels: list[LabelInfo] = []  # 标签信息，不一定按照标签顺序（LabelValue）排列
        return

    def SetLabelInfos(self, labels: List[LabelInfo]) -> None:
        self.labels = labels

    def SetPath(self, folderPath: str) -> None:
        self.folderPath = folderPath
        self.labels: List[LabelInfo] = []
        # 获取文件夹下的所有文件夹
        folderList = [folder for folder in os.listdir(folderPath) if (CONST_NeedTrain in folder and '$' in folder)]
        # 遍历文件夹提取文件夹名称
        for index, folder in enumerate(folderList):
            # folder 格式如 主分类名称$子分类编号_NeedTrain_BigPic
            # 获取主分类名称，从文件名中删除 CONST_NeedTrain，并根据 $ 分割
            [name1, subIndex] = folder.replace(CONST_NeedTrain, '').replace(CONST_BigPic, '').split('$')
            self.labels.append(LabelInfo(name1, subIndex, index, os.path.join(folderPath, folder)))

    def SaveToCsv(self, savePath: str) -> None:
        ciContent = "LabelName,LabelValue,LabelPath,MainName,SubIndex\n"
        for label in self.labels:
            # ciContent += '{}{} {} {} {} {}\n'.format(name1, subIndex, index, name1, int(subIndex), folder)
            ciContent += '{},{},{},{},{}\n'.format(label.label_name, label.label_value, label.label_dir_path, label.label_main_name, label.label_sub_index_name)
        with open(savePath, 'w') as f:
            f.write(ciContent)

    @staticmethod
    def ReadFromCsv(filePath: str) -> 'DatasetAnalyser':
        '''
        反序列化 csv 文件
        '''
        labels: List[LabelInfo] = []
        with open(filePath, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.replace('\n', '')
                # 分割字符串
                [label_name, label_value, label_dir_path, label_main_name, label_sub_index_name] = line.split(',')
                labels.append(LabelInfo(label_main_name, label_sub_index_name, int(label_value), label_dir_path))
        ret = DatasetAnalyser()
        ret.SetLabelInfos(labels)
        return ret


if __name__ == '__main__':
    # 分析指定文件夹下的文件夹名称，生成一个配置文件
    analyser = DatasetAnalyser()
    analyser.SetPath('C:\\Unpack\\qc_train')
    analyser.SaveToCsv('C:\\Unpack\\qc_train\\label_info.csv')
    a2 = DatasetAnalyser.ReadFromCsv("test.csv")
