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
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo
from ClassifyTraining_Settings import ClassifyTraining_Settings
from ClassifyTraining_Dataset import ClassifyTraining_Dataset
from tensorboardX import SummaryWriter
from ClassifyTraining_Trainer import ITrainer
from ClassifyTraining_Trainer_MobileNetV3 import mobilenet_v3_large


class CustomHardswish(nn.Module):
    """
    Export-friendly version of nn.Hardswish()
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


def _set_module(model, submodule_key, module):
    # 核心函数，参考了torch.quantization.fuse_modules()的实现
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def GetMobileNetV3(label_count: int):
    # 建立模型
    model = mobilenet_v3_large(pretrained=True)  # 加载修改过算子的 mobilenet v3 预训练模型
    # model = models.mobilenet_v3_large(pretrained=True)
    # # 替换 ONNX 不支持的算子(可选)
    # hs = []
    # for k, m in model.named_modules():
    #     if isinstance(m, nn.Hardswish):
    #         hs.append(k)
    #     pass
    # for k in hs:
    #     _set_module(model, k, CustomHardswish())
    # 修改输出层
    nn0_in = model.classifier[0].in_features
    nn0_out = model.classifier[0].out_features
    model.classifier = nn.Sequential(
        nn.Linear(nn0_in, nn0_out),
        nn.Hardswish(),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(nn0_out, label_count),
    )
    return model


class MobileNetV3Trainer(ITrainer):

    @abstractmethod
    def init_model(self) -> nn.Module:
        '''
        初始化一个新的模型
        '''
        self.logger.info("Model Init: class num = {}".format(self.Settings.LabelCount))
        return GetMobileNetV3(self.Settings.LabelCount)

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
    # s = ClassifyTraining_Settings()
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
    train_settings_json_path = ""
    epoch = -1
    test_image_path = ""
    try:
        opts, args = getopt.getopt(argv, "-h-i:-e:-t:", ["i=", "t="])
        if len(opts) == 0:
            print('please set args: -i <train settings json path>')
            sys.exit()
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <train settings json path>')
                    sys.exit()
                if opt in ("-i"):
                    train_settings_json_path = arg
                    if os.path.exists(train_settings_json_path) == False:
                        print('train settings json file not exist: {}'.format(train_settings_json_path))
                        sys.exit(1)
                if opt in ("-e"):
                    epoch = int(arg)
                if opt in ("-t"):
                    test_image_path = arg
                    if os.path.exists(test_image_path) == False:
                        print('test image file not exist: {}'.format(test_image_path))
                        sys.exit(1)
    except getopt.GetoptError:
        print('please set args: -i <train settings json path>')
        sys.exit(2)

    t = MobileNetV3Trainer(train_settings_json_path)
    if test_image_path == "":
        t.train()
    else:
        t.test(test_image_path, epoch)
