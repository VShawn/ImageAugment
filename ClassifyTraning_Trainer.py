import sys
import getopt
import os
import json
import re
import time
import logging
import torch
import turtle
from torch import nn, Tensor, optim
from torchvision import models
from torch.utils.data import Dataset as TorchDataSet, DataLoader
from ClassifyPreprocess_DataSetAnalyser import DataSetAnalyser, LabelInfo
from ClassifyTraning_TrainingSettings import TrainingSettings


class ITrainer(object):
    def __init__(self, settings: TrainingSettings) -> None:
        settings.Verdiate()
        self.Settings: TrainingSettings = settings
        self.Model: nn.Module = None
        self.Optimizer: torch.optim.Optimizer = None
        self.Loss: nn.modules.loss._Loss = None
        self.__basic_init_logger()
        pass

    def __basic_init_logger(self):
        # 创建保存文件夹
        if not os.path.exists(self.outDirPath):
            os.makedirs(self.outDirPath)
        # 日志写到当前时间的文件中
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        # 使用FileHandler输出到文件
        fh = logging.FileHandler(os.path.join(self.outDirPath, '{}_{}.log'.format(time.strftime('%Y%m%d_%H%M%S'), os.path.basename(__file__))))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        # 添加两个Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def save_checkpoint(self, is_best: bool, epoch: int) -> None:
        if self.Settings.UseGpuCount > 1:
            checkpoint = {'model': self.Model.module,
                          'model_state_dict': self.Model.module.state_dict(),
                          'optimizer_state_dict': self.Optimizer.state_dict(),
                          'epoch': epoch}
        else:
            checkpoint = {'model': self.Model,
                          'model_state_dict': self.Model.state_dict(),
                          'optimizer_state_dict': self.Optimizer.state_dict(),
                          'epoch': epoch}
        torch.save(checkpoint, os.path.join(self.Settings.OutPutPath, 'epoch_{}.pth'.format(epoch)))
        if is_best:
            torch.save(checkpoint, os.path.join(self.Settings.OutPutPath, 'best.pth'))

    @staticmethod
    def resume_model(settings: TrainingSettings, logger: logging.Logger) -> tuple[nn.Module, optim.Optimizer]:
        '''
        从 checkpoint 加载模型
        '''
        # TODO checkpoint 的读取方法
        model = torch.load(settings.ResumeFrom)
        if model is None:
            logger.error("ResumeFrom {} is not exist".format(settings.ResumeFrom))
            exit(1)
        return model

    @staticmethod
    def init_model(settings: TrainingSettings, logger: logging.Logger) -> tuple[nn.Module, optim.Optimizer]:
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
            nn.Linear(nn0_out, settings.classNum),
        )
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=settings.LR)
        # optimizer = optim.SGD(model.parameters(), lr=settings.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        return model, optimizer

    def init_loss(self) -> nn.modules.loss._Loss:
        # 损失函数
        loss = nn.CrossEntropyLoss()
        # loss = nn.BCEWithLogitsLoss()
        # loss = LabelSmoothSoftmaxCE()
        # loss = LabelSmoothingCrossEntropy()
        return loss

    def train(self, saveToDir: str) -> None:
        '''
        开始训练
        从几个虚函数中读取网络参数，然后根据配置开始训练
        '''
        # 创建训练模型参数保存的文件夹
        if not os.path.exists(saveToDir):
            os.makedirs(saveToDir)
        # 初始化模型
        if self.Settings.ResumeFrom == "":
            self.Model, self.Optimizer = self.init_model(self.Settings, self.logger)
        else:
            self.Model, self.Optimizer = self.resume_model(self.Settings, self.logger)

        # 配置 pytorch 环境
        if self.Settings.UseGpuCount == 0:
            # 使用CPU训练
            self.logger.info("Using CPU")
        else:
            # 使用 GPU 训练
            self.logger.info("Using GPU")
            if torch.cuda.is_available() == False:
                self.logger.error("GPU is not available")
                exit(1)
            # 多 GPU 训练
            if self.Settings.UseGpuCount > 1:
                self.Model = nn.DataParallel(self.Model, device_ids=list(range(self.Settings.UseGpuCount)))
            self.Model.cuda()

        # 训练前参数确认
        lr = cfg.LR
        batch_size = cfg.BATCH_SIZE
        # 每一个epoch含有多少个batch
        max_batch = len(train_datasets)//batch_size
        epoch_size = len(train_datasets) // batch_size
        # 训练max_epoch个epoch
        max_iter = cfg.MAX_EPOCH * epoch_size
        start_iter = cfg.RESUME_EPOCH * epoch_size
        epoch = cfg.RESUME_EPOCH
        # cosine学习率调整
        warmup_epoch = 5
        warmup_steps = warmup_epoch * epoch_size
        global_step = 0
        # step 学习率调整参数
        stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
        step_index = 0
        model.train()


if __name__ == '__main__':
    model = models.mobilenet_v3_large()
    print(model)
