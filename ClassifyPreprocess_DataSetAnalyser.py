import os
import json
from dataclasses import dataclass

CONST_NeedTrain = "_NeedTrain"
CONST_BigPic = "_BigPic"

class ClassifyDataFolderAnalyser(object):
    """
    分析指定数据集文件夹下的文件夹名称，生成一个标签配置文件
    """
    def __init__(self) -> None:
        pass

    def analysis(self, folderPath: str, savePath: str) -> None:
        # 获取文件夹下的所有文件夹
        folderList = os.listdir(folderPath)
        folderList = [folder for folder in folderList if (CONST_NeedTrain in folder and '$' in folder)]
        ciContent = ""
        # 遍历文件夹提取文件夹名称
        for index, folder in enumerate(folderList):
            # folder 格式如 主分类名称$子分类编号_NeedTrain_BigPic

            # 获取主分类名称，从文件名中删除 CONST_NeedTrain，并根据 $ 分割
            [name1, subIndex] = folder.replace(CONST_NeedTrain, '').replace(CONST_BigPic, '').split('$')
            # ciContent += '{}{} {} {} {} {}\n'.format(name1, subIndex, index, name1, int(subIndex), folder)
            ciContent += '{}{} {}\n'.format(name1, subIndex, index)

        # 保存文件
        with open(savePath, 'w') as f:
            f.write(ciContent)

if __name__ == '__main__':
    # 分析指定文件夹下的文件夹名称，生成一个配置文件
    analyser = ClassifyDataFolderAnalyser()
    analyser.analysis('E:\BM3000-TEST\B\FiveCells', "test.ci")