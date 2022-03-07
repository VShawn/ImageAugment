import os
import json
from dataclasses import dataclass
import struct

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
        self.image_paths = [os.path.join(label_dir_path, file) for file in os.listdir(label_dir_path) if (file.lower().endswith('.bmp') or file.lower().endswith('.jpg') or file.lower().endswith('.png'))]
        # 文件夹内的 bmp jpg 图片数量
        self.image_count = len(self.image_paths)

class DataSetAnalyser(object):
    """
    分析指定数据集文件夹下的文件夹名称，生成一个标签配置文件
    """
    def __init__(self, folderPath: str):
        self.SetPath(folderPath)
        return

    def SetPath(self, folderPath: str) -> None:
        self.folderPath = folderPath
        self.labels = []
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

if __name__ == '__main__':
    # 分析指定文件夹下的文件夹名称，生成一个配置文件
    analyser = DataSetAnalyser('E:\BM3000-TEST\B\FiveCells')
    analyser.SaveToCsv("test.csv")