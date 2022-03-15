from abc import abstractmethod
import sys
import getopt
import os
import json
import time
import logging
import torch
from types import SimpleNamespace
from torch import nn, Tensor, optim
from torchvision import models
from torch.utils.data import Dataset as TorchDataset, DataLoader
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo
from ClassifyTraning_Settings import ClassifyTraning_Settings
from ClassifyTraning_Dataset import ClassifyTraning_Dataset
from tensorboardX import SummaryWriter
from ClassifyTraning_Trainer import ITrainer


class MobileNetV3Trainer(ITrainer):

    @abstractmethod
    def init_model(self) -> nn.Module:
        '''
        初始化一个新的模型
        '''
        # 建立模型
        model = models.mobilenet_v3_large(pretrained=True)
        # 修改输出层
        nn0_in = model.classifier[0].in_features
        nn0_out = model.classifier[0].out_features
        model.classifier = nn.Sequential(
            nn.Linear(nn0_in, nn0_out),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(nn0_out, self.Settings.LabelCount),
        )
        self.logger.info("Model Init: {}, class num = {}".format(model, self.Settings.LabelCount))
        return model

    @abstractmethod
    def init_optimizer(self) -> optim.Optimizer:
        '''
        初始化优化器
        '''
        # 优化器
        optimizer = optim.Adam(self.Model.parameters(), lr=self.Settings.LR)
        # optimizer = optim.SGD(self.Model.parameters(), lr=self.Settings.LR, momentum=self.Settings.MOMENTUM, weight_decay=self.Settings.WEIGHT_DECAY)
        return optimizer

    @abstractmethod
    def init_loss_function(self) -> nn.modules.loss._Loss:
        '''
        初始化损失函数
        '''
        loss_function = nn.CrossEntropyLoss()
        # loss_function = nn.BCEWithLogitsLoss()
        # loss_function = LabelSmoothSoftmaxCE()
        # loss_function = LabelSmoothingCrossEntropy()
        return loss_function

    @abstractmethod
    def optimizer_lr_adjust(self, learning_rate_base: float, current_epoch: int) -> None:
        if self.LrUpdater is None:
            # self.LrUpdater = torch.optim.lr_scheduler.StepLR(self.Optimizer, step_size=5, gamma=0.1, last_epoch=current_epoch - 1)
            # self.LrUpdater = torch.optim.lr_scheduler.ExponentialLR(self.Optimizer, gamma=0.1, last_epoch=current_epoch - 1)
            self.LrUpdater = torch.optim.lr_scheduler.CosineAnnealingLR(self.Optimizer, T_max=10, last_epoch=current_epoch - 1)
            # self.LrUpdater = torch.optim.lr_scheduler.ReduceLROnPlateau(self.Optimizer)
        self.LrUpdater.step()


if __name__ == '__main__':
    # s = ClassifyTraning_Settings()
    # s.set_dataset_path(r'D:\UritWorks\AI\image_preprocess\Augmented')
    # if os.path.exists(s.OutputDirPath) == False:
    #     os.makedirs(s.OutputDirPath)
    # setting_path = os.path.join(s.OutputDirPath, 'demo_train_setting.json')
    # s.to_json_file(setting_path)
    # t = ITrainer(setting_path)
    # t.train()
    # t = MobileNetV3Trainer(r'D:\UritWorks\AI\image_preprocess\DemoTrained\demo_train_setting.json_20220311113655.json')
    # t.train()

    # 读取输入参数
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "-h-i::", ["i="])
        if len(opts) == 0:
            print('please set args: -i <train settings json path>')
            sys.exit()
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <train settings json path>')
                    sys.exit()
                else:
                    if opt in ("-i"):
                        inputJson = arg
                        if os.path.exists(inputJson) == False:
                            print('train settings json file not exist: {}'.format(inputJson))
                            sys.exit(1)
    except getopt.GetoptError:
        print('please set args: -i <train settings json path>')
        sys.exit(2)

    t = MobileNetV3Trainer(inputJson)
    t.train()
