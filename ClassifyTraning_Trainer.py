from abc import abstractmethod
import sys
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
from abc import ABC, abstractmethod


class ITrainer(object):
    def __init__(self, settings_json_path: str) -> None:
        self.Settings: ClassifyTraning_Settings = ClassifyTraning_Settings.from_json_file(settings_json_path)
        self.settings_json_path = settings_json_path
        assert(self.Settings is not None)
        self.Settings.verdiate()
        self.Model: nn.Module = None
        self.Optimizer: torch.optim.Optimizer = None
        self.LossFunction: nn.modules.loss._Loss = None
        self.LrUpdater = None
        self.train_loader: DataLoader = None
        self.eval_loader: DataLoader = None
        self.BestLoss = sys.float_info.max
        pass

    def __basic_init_logger(self):
        # 创建保存文件夹
        if not os.path.exists(self.Settings.OutputDirPath):
            os.makedirs(self.Settings.OutputDirPath)
        # 日志写到当前时间的文件中
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        # 使用FileHandler输出到文件
        fh = logging.FileHandler(os.path.join(self.Settings.OutputDirPath, '{}_{}_run_log.log'.format(self.Settings.ProjectName, self.Settings.TrainingId)))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        # 添加两个Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    # def _get_checkpoint_path(self, epoch: int) -> str:
    #     dir = os.path.join(self.Settings.OutputDirPath, '{}_checkpoints'.format(self.Settings.TrainingId))
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     return os.path.join(dir, 'epoch_%d.pth' % epoch)

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
        torch.save(checkpoint, self.Settings.get_checkpoint_path(epoch))
        if is_best:
            # 保存一个单一完整模型
            torch.save(self.Model, os.path.join(self.Settings.OutputDirPath, '{}_{}_best.pth').format(self.Settings.ProjectName, self.Settings.TrainingId))

    def load_checkpoint(self, epoch: int) -> nn.Module:
        '''
        从 checkpoint 加载模型
        '''
        path = self.Settings.get_checkpoint_path(epoch)
        checkpoint = torch.load(path)
        if checkpoint is None:
            self.logger.error("ResumeFrom {} is not exist".format(epoch))
            exit(1)
        self.logger.info("Model Resumed from: {}".format(path))
        model = checkpoint['model']  # 提取网络结构
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
        return model

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
        # """Sets the learning rate
        # # Adapted from PyTorch Imagenet example:
        # # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        # """
        # gamma: float = 0.1  # 学习率衰减系数
        # if current_epoch < 4:
        #     lr = 1e-6 + (learning_rate_base-1e-6) * iteration / (epoch_size * 5)
        # else:
        #     lr = learning_rate_base * (gamma ** (step_index))
        # for param_group in self.Optimizer.param_groups:
        #     param_group['lr'] = lr
        # return lr

    def _train_or_eval_one_epoch(self, current_epoch: int, is_train: bool) -> None:
        '''
        执行一次 epoch 的训练或测试
        '''
        write_to_session = 'Train' if is_train else 'Eval'
        loader = self.train_loader if is_train else self.eval_loader
        batch_count = len(loader)
        if is_train:
            self.Model.train()
        else:
            self.Model.eval()
        loss: float = 0.0
        acc_top1: float = 0.0
        acc_top1_90: float = 0.0
        acc_top5: float = 0.0
        # 读取一遍 loader ，完成一次 epoch 训练
        for i, (labels, images) in enumerate(loader):
            if i % 32 == 0:
                self.logger.debug('epoch: {}, {} progress {} / {}'.format(current_epoch, write_to_session, i, batch_count))
            if self.Settings.UseGpuCount > 0:
                labels = labels.cuda()
                images = images.cuda()
            # forward
            if is_train:
                self.Optimizer.zero_grad()
                outputs = self.Model(images)
                tensor_loss = self.LossFunction(outputs, labels)
            else:
                with torch.no_grad():
                    outputs = self.Model(images)
                    tensor_loss = self.LossFunction(outputs, labels)
            # backward
            if is_train:
                tensor_loss.backward()
                self.Optimizer.step()
                # 更新 LR
                self.optimizer_lr_adjust(self.Settings.LR, current_epoch)
            # 统计
            loss += tensor_loss.item()
            for i in range(len(labels)):
                percentages, indices = torch.sort(outputs[i], descending=True)
                percentages = percentages.softmax(0)
                for j in range(5):
                    if labels[i] == indices[j]:
                        if j == 0:
                            acc_top1 += 1
                            top1_p = percentages[0].item()
                            if top1_p > 0.9:
                                acc_top1_90 += 1
                        acc_top5 += 1
                        break
        # 计算平均值
        loss = loss / batch_count
        acc_top1 = acc_top1 / batch_count
        acc_top1_90 = acc_top1_90 / batch_count
        acc_top5 = acc_top5 / batch_count
        # 打印
        self.logger.info('epoch: %d, loss: %.4f, acc_top1: %.4f, acc_top1_90: %.4f, acc_top5: %.4f' % (current_epoch, loss, acc_top1, acc_top1_90, acc_top5))
        with SummaryWriter(os.path.join(self.Settings.OutputDirPath, '{}_{}_train_log/loss'.format(self.Settings.ProjectName, self.Settings.TrainingId))) as sw_loss, \
                SummaryWriter(os.path.join(self.Settings.OutputDirPath, '{}_{}_train_log/acc_top1'.format(self.Settings.ProjectName, self.Settings.TrainingId))) as sw_acc_top1, \
                SummaryWriter(os.path.join(self.Settings.OutputDirPath, '{}_{}_train_log/acc_top1_90'.format(self.Settings.ProjectName, self.Settings.TrainingId))) as sw_acc_top1_90, \
                SummaryWriter(os.path.join(self.Settings.OutputDirPath, '{}_{}_train_log/acc_top5'.format(self.Settings.ProjectName, self.Settings.TrainingId))) as sw_acc_top5, \
                SummaryWriter(os.path.join(self.Settings.OutputDirPath, '{}_{}_train_log/LR'.format(self.Settings.ProjectName, self.Settings.TrainingId))) as sw_lr:
            sw_loss.add_scalar(write_to_session, loss, global_step=current_epoch)
            sw_acc_top1.add_scalar(write_to_session, acc_top1, global_step=current_epoch)
            sw_acc_top1_90.add_scalar(write_to_session, acc_top1_90, global_step=current_epoch)
            sw_acc_top5.add_scalar(write_to_session, acc_top5, global_step=current_epoch)
            sw_lr.add_scalar(write_to_session, self.Optimizer.param_groups[0]['lr'], global_step=current_epoch)

        # 保存模型
        if is_train == False:
            is_best = False
            # 保存训练进度
            self.Settings.ResumeEpoch = current_epoch
            self.Settings.ResumeLR = self.Optimizer.param_groups[0]['lr']
            self.Settings.to_json_file(self.settings_json_path)
            if loss < self.BestLoss:
                is_best = True
                self.BestLoss = loss
            self.save_checkpoint(is_best, current_epoch)

    def train(self) -> None:
        '''
        开始训练
        从几个虚函数中读取网络参数，然后根据配置开始训练
        '''
        self.Settings.verdiate()
        self.Settings.start_new_train_if_not_resume()
        self.settings_json_path = '{}_{}.json'.format(self.settings_json_path, self.Settings.TrainingId)
        self.__basic_init_logger()

        # 创建训练模型参数保存的文件夹
        if not os.path.exists(self.Settings.OutputDirPath):
            os.makedirs(self.Settings.OutputDirPath)

        # 初始化模型
        self.Model = self.load_checkpoint(self.Settings.ResumeEpoch) if self.Settings.ResumeEpoch > 0 else self.init_model()
        self.Optimizer = self.init_optimizer()
        self.LossFunction = self.init_loss_function()

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

        # 创建 loader
        self.train_loader, self.eval_loader = ClassifyTraning_Dataset.get_train_validate_data_loader(label_info_csv_path=self.Settings.DatasetLabelInfoCsvPath,
                                                                                                     input_image_size=self.Settings.InputSize,
                                                                                                     batch_size=self.Settings.BatchSize,
                                                                                                     num_workers=4,
                                                                                                     validate_ratio=0.2)
        self.logger.info("train_loader items count = {}".format(len(self.train_loader)))
        self.logger.info("eval_loader items count = {}".format(len(self.eval_loader)))

        # 开始训练
        for current_epoch in range(self.Settings.ResumeEpoch, self.Settings.Epochs):
            self.logger.info('epoch {} start'.format(current_epoch))
            self._train_or_eval_one_epoch(current_epoch, True)
            self._train_or_eval_one_epoch(current_epoch, False)
            pass


if __name__ == '__main__':
    s = ClassifyTraning_Settings()
    s.set_dataset_path(r'D:\UritWorks\AI\image_preprocess\Augmented')
    if os.path.exists(s.OutputDirPath) == False:
        os.makedirs(s.OutputDirPath)
    setting_path = os.path.join(s.OutputDirPath, 'demo_train_setting.json')
    s.to_json_file(setting_path)
    t = ITrainer(setting_path)
    t.train()
