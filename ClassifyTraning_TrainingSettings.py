import sys
import getopt
import os
import json
import re
import time
import logging
from ClassifyPreprocess_DataSetAnalyser import DataSetAnalyser, LabelInfo


class TrainingSettings:
    '''
    单次训练参数配置类，将被保存到 checkpoint 文件同目录
    '''

    def __init__(self) -> None:
        self.DataSetPath = ""
        self.LabelCount = 0  # 类别数量
        self.InputSize = (224, 224)  # 输入图像大小
        self.BatchSize = 32  # 批次大小
        self.Epochs = 10  # 迭代次数
        self.LR = 0.001  # 初始学习率
        self.ResumeFrom = ""  # 恢复训练的模型路径
        self.ResumeLR = 0.001  # 恢复训练时的学习率
        self.UseGpuCount = 1  # 用几个GPU训练，0 则用 CPU 训练
        self.OutPutPath = ""  # 输出模型路径

    def SetDataSetPath(self, dataSetPath: str):
        '''
        设置数据集位置
        '''
        self.DataSetPath = dataSetPath
        self.Verdiate()
        pass

    def Verdiate(self):
        '''
        验证配置是否正确，有问题会抛出异常，每次训练前建议都调用检查一下
        '''
        if self.DataSetPath == "":
            raise Exception("DataSetPath is empty")
        # 确认数据集存在
        assert(os.path.isdir(self.DataSetPath))
        assert(os.path.exists(os.path.join(self.DataSetPath, 'label_info.csv')))
        # 确认数据集中标签数量正确
        dataSetAnalyser = DataSetAnalyser.ReadFromCsv(os.path.join(self.DataSetPath, 'label_info.csv'))
        self.LabelCount = len(dataSetAnalyser.labels)
        assert(self.outPutPath != "")
        assert(self.LabelCount > 0)
        assert(self.BatchSize > 0)
        assert(self.LR > 0)
        assert(self.Epochs > 0)
        if self.ResumeFrom != "":
            assert(os.path.exists(self.ResumeFrom))
            assert(self.ResumeLR > 0)
        pass

    def ToDict(self):
        return self.__dict__

    @staticmethod
    def FromDict(settings: dict) -> 'TrainingSettings':
        ret = TrainingSettings()
        for key in ret.__dict__:
            if key in settings:
                ret.__dict__[key] = settings[key]
        ret.Verdiate()
        return ret

    def ToJson(self) -> str:
        return json.dumps(self, default=lambda o: o.ToDict(), indent=4, separators=(',', ': '))

    def ToJsonFile(self, filePath):
        with open(filePath, 'w') as f:
            f.write(self.ToJson())
        pass

    @staticmethod
    def FromJson(strJson: str) -> 'TrainingSettings':
        jobj = json.loads(strJson)
        return TrainingSettings.FromDict(jobj)

    @staticmethod
    def FromJsonFile(strJson: str) -> 'TrainingSettings':
        with open(strJson, 'r') as f:
            return TrainingSettings.FromJson(f.read())


if __name__ == '__main__':
    settings = TrainingSettings()
    settings.SetDataSetPath('Augmented')
    print(settings.ToDict())
    print(settings.ToJson())
    print(TrainingSettings.FromJson(settings.ToJson()).ToJson())
