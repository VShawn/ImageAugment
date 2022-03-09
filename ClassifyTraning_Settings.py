import sys
import getopt
import os
import json
import re
import time
import logging
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo


class ClassifyTraning_Settings:
    '''
    单次训练参数配置类，将被保存到 checkpoint 文件同目录
    '''

    def __init__(self) -> None:
        self.ProjectName: str = "Model"
        self.DatasetPath: str = ""
        self.DatasetLabelInfoCsvPath: str = ""
        self.LabelCount: int = 0  # 类别数量
        self.InputSize: int = 224  # 输入图像大小
        self.BatchSize: int = 32  # 批次大小，每次喂给模型的样本数量。
        self.Epochs: int = 10  # 迭代轮数
        self.LR: float = 0.001  # 初始学习率
        self.ResumeEpoch: int = 0  # 恢复从哪个 Epoch 恢复训练
        self.ResumeLR: float = 0.001  # 恢复训练时的学习率
        self.UseGpuCount: int = 1  # 用几个GPU训练，0 则用 CPU 训练
        self.OutputDirPath: str = "Out"  # 输出模型路径

    def SetDatasetPath(self, dataSetPath: str):
        '''
        设置数据集位置
        '''
        self.DatasetPath = dataSetPath
        self.DatasetPath = os.path.join(self.DatasetPath, 'label_info.csv')
        self.Verdiate()
        pass

    def Verdiate(self):
        '''
        验证配置是否正确，有问题会抛出异常，每次训练前建议都调用检查一下
        '''
        if self.DatasetPath == "":
            raise Exception("DatasetPath is empty")
        # 确认数据集存在
        assert(os.path.isdir(self.DatasetPath))
        assert(os.path.exists(self.DatasetLabelInfoCsvPath))
        # 确认数据集中标签数量正确
        dataSetAnalyser = DatasetAnalyser.ReadFromCsv(os.path.join(self.DatasetPath, 'label_info.csv'))
        self.LabelCount = len(dataSetAnalyser.labels)
        assert(self.outPutPath != "")
        assert(self.LabelCount > 0)
        assert(self.LabelCount > 0)
        assert(self.BatchSize > 0)
        assert(self.LR > 0)
        assert(self.Epochs > 0)
        if self.ResumeFrom != "":
            assert(os.path.exists(self.ResumeFrom))
            assert(self.ResumeLR > 0)
        pass

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(settings: dict) -> 'ClassifyTraning_Settings':
        ret = ClassifyTraning_Settings()
        for key in ret.__dict__:
            if key in settings:
                ret.__dict__[key] = settings[key]
        ret.Verdiate()
        return ret

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.to_dict(), indent=4, separators=(',', ': '))

    def to_json_file(self, filePath):
        with open(filePath, 'w') as f:
            f.write(self.to_json())
        pass

    @staticmethod
    def from_json(strJson: str) -> 'ClassifyTraning_Settings':
        jobj = json.loads(strJson)
        return ClassifyTraning_Settings.from_dict(jobj)

    @staticmethod
    def from_json_file(strJson: str) -> 'ClassifyTraning_Settings':
        with open(strJson, 'r') as f:
            return ClassifyTraning_Settings.from_json(f.read())


if __name__ == '__main__':
    settings = ClassifyTraning_Settings()
    settings.SetDatasetPath('Augmented')
    print(settings.to_dict())
    print(settings.to_json())
    print(ClassifyTraning_Settings.from_json(settings.to_json()).to_json())
