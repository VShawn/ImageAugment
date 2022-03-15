from posixpath import isabs
import sys
import getopt
import os
import json
import re
import time
import logging

from httpx import patch
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo


class ClassifyTraning_Settings:
    '''
    单次训练参数配置类，将被保存到 checkpoint 文件同目录
    '''

    def __init__(self) -> None:
        self.ProjectName: str = "ModelForDemo"
        self.TrainingId: str = None  # 以当前时间作为本次训练Id与文件名，每次运行训练时，都会生成一个新的Id
        self.DatasetPath: str = ""
        self.DatasetLabelInfoCsvPath: str = ""
        self.InputSize: int = 224  # 输入图像大小
        self.BatchSize: int = 32  # 批次大小，每次喂给模型的样本数量。
        self.Epochs: int = 1000  # 迭代轮数
        self.SaveModelEpoch = 1  # 模型保存间隔
        self.LR: float = 0.01  # 初始学习率
        self.UseGpuCount: int = 1  # 用几个GPU训练，0 则用 CPU 训练
        self.OutputDirPath: str = "DemoTrained"  # 输出模型路径
        self.ResumeEpoch: int = 0  # 恢复从哪个 Epoch 恢复训练
        self.ResumeLR: float = 0.01  # 恢复训练时的学习率

    def get_checkpoint_path(self, epoch: int):
        dir = os.path.join(self.OutputDirPath, '{}_{}_checkpoints'.format(self.ProjectName, self.TrainingId))
        if not os.path.exists(dir):
            os.makedirs(dir)
        return os.path.join(dir, 'epoch_%d.pth' % epoch)

    def set_dataset_path(self, dataSetPath: str):
        '''
        设置数据集位置
        '''
        self.DatasetPath = dataSetPath
        if not os.path.isabs(self.DatasetPath):
            self.DatasetPath = os.path.abspath(self.DatasetPath)
        self.DatasetLabelInfoCsvPath = os.path.join(self.DatasetPath, 'label_info.csv')
        self.verdiate()
        pass

    def start_new_train_if_not_resume(self):
        '''
        开始新的训练，生成新的训练Id
        '''
        if self.ResumeEpoch > 0:
            assert(self.ResumeLR > 0)
        else:
            # 生成新的训练Id
            self.TrainingId = time.strftime("%Y%m%d%H%M%S", time.localtime())
        pass

    def verdiate(self):
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
        assert(self.OutputDirPath != "")
        assert(self.LabelCount > 0)
        assert(self.LabelCount > 0)
        assert(self.BatchSize > 0)
        assert(self.LR > 0)
        assert(self.Epochs > 0)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(settings: dict) -> 'ClassifyTraning_Settings':
        ret = ClassifyTraning_Settings()
        for key in ret.__dict__:
            if key in settings:
                ret.__dict__[key] = settings[key]
        ret.verdiate()
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
    # # test code
    # settings = ClassifyTraning_Settings()
    # settings.set_dataset_path('Augmented')
    # print(settings.to_dict())
    # print(settings.to_json())
    # print(ClassifyTraning_Settings.from_json(settings.to_json()).to_json())
    # 读取输入参数
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "-h-i:-o:-c:", ["i=", "o=", "c="])
        if len(opts) == 0:
            print('please set args: -i <augmented dataset dir path> -o <a json file path>')
            sys.exit()
            # inputDir = 'E:\BM3000-TEST\B\FiveCells'
            # outputDir = 'Augmented'
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <augmented dataset dir path> -o <a json file path>')
                    sys.exit()
                else:
                    if opt in ("-i"):
                        inputDir = arg
                    elif opt in ("-o"):
                        outputJsonFile = arg
    except getopt.GetoptError:
        print('please set args: -i <augmented dataset dir path> -o <a json file path>')
        sys.exit(2)

    if not os.path.isabs(outputJsonFile):
        outputJsonFile = os.path.abspath(outputJsonFile)
    dir_path = os.path.dirname(outputJsonFile)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    settings = ClassifyTraning_Settings()
    settings.OutputDirPath = dir_path
    settings.set_dataset_path(inputDir)
    settings.to_json_file(outputJsonFile)
    print('we generate a json file: {}'.format(outputJsonFile))
