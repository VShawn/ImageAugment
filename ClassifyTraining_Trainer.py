from abc import abstractmethod
import sys
import os
import json
import time
import logging
from tkinter.messagebox import NO
import torch
from types import SimpleNamespace
from torch import nn, Tensor, optim
from torchvision import models
from torch.utils.data import Dataset as TorchDataset, DataLoader
from ClassifyPreprocess_DatasetAnalyser import DatasetAnalyser, LabelInfo
from ClassifyTraining_Settings import ClassifyTraining_Settings
from ClassifyTraining_Dataset import ClassifyTraining_Dataset
from tensorboardX import SummaryWriter


class ITrainer(object):
    def __init__(self, settings_json_path: str) -> None:
        self.Settings: ClassifyTraining_Settings = ClassifyTraining_Settings.from_json_file(settings_json_path)
        self.settings_json_path = settings_json_path
        assert(self.Settings is not None)
        self.Settings.verdiate()
        self.Model: nn.Module = None
        self.Optimizer: torch.optim.Optimizer = None
        self.loss_func: nn.modules.loss._Loss = None
        self.LrUpdater: torch.optim.lr_scheduler._LRScheduler = None
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
                          #   'optimizer_state_dict': self.Optimizer.state_dict(),
                          'epoch': epoch}
        else:
            checkpoint = {'model': self.Model,
                          'model_state_dict': self.Model.state_dict(),
                          #   'optimizer_state_dict': self.Optimizer.state_dict(),
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

    def load_best_pth(self):
        '''
        加载最好的模型
        '''
        best_path = os.path.join(self.Settings.OutputDirPath, '{}_{}_best.pth').format(self.Settings.ProjectName, self.Settings.TrainingId)
        if not os.path.exists(best_path):
            self.logger.error("ResumeFrom {} is not exist".format(epoch))
        model = torch.load(best_path)
        if model is None:
            self.logger.error("ResumeFrom {} is not exist".format(epoch))
            exit(1)
        return model

    @abstractmethod
    def init_model(self) -> nn.Module:
        '''
        初始化一个新的模型，并返回
        '''
        # # 建立模型
        # model = models.mobilenet_v3_large(pretrained=True)
        # # 修改输出层
        # nn0_in = model.classifier[0].in_features
        # nn0_out = model.classifier[0].out_features
        # model.classifier = nn.Sequential(
        #     nn.Linear(nn0_in, nn0_out),
        #     nn.Hardswish(),
        #     nn.Dropout(0.2, inplace=True),
        #     nn.Linear(nn0_out, self.Settings.LabelCount),
        # )
        # self.logger.info("Model Init: {}, class num = {}".format(model, self.Settings.LabelCount))
        # return model
        raise NotImplemented

    @abstractmethod
    def init_optimizer(self) -> optim.Optimizer:
        '''
        初始化一个优化器，并返回
        '''
        # # 优化器
        # optimizer = optim.Adam(self.Model.parameters(), lr=self.Settings.LR)
        # # optimizer = optim.SGD(self.Model.parameters(), lr=self.Settings.LR, momentum=self.Settings.MOMENTUM, weight_decay=self.Settings.WEIGHT_DECAY)
        # return optimizer
        raise NotImplemented

    @abstractmethod
    def init_loss_function(self) -> nn.modules.loss._Loss:
        '''
        初始化一个损失函数，并返回
        '''
        # loss_function = nn.CrossEntropyLoss()
        # # loss_function = nn.BCEWithLogitsLoss()
        # # loss_function = LabelSmoothSoftmaxCE()
        # # loss_function = LabelSmoothingCrossEntropy()
        # return loss_function
        raise NotImplemented

    @abstractmethod
    def optimizer_lr_adjust(self, learning_rate_base: float, current_epoch: int) -> None:
        '''
        每次调用都调节一次 LR
        '''
        # # 使用 lr_scheduler，第一次调用时初始化一个 LR 调节器，后续调用时，根据 current_epoch 调整 self.Optimizer 的 LR
        # if self.LrUpdater is None:
        #     # self.LrUpdater = torch.optim.lr_scheduler.StepLR(self.Optimizer, step_size=5, gamma=0.1, last_epoch=current_epoch - 1)
        #     # self.LrUpdater = torch.optim.lr_scheduler.ExponentialLR(self.Optimizer, gamma=0.1, last_epoch=current_epoch - 1)
        #     self.LrUpdater = torch.optim.lr_scheduler.CosineAnnealingLR(self.Optimizer, T_max=10, last_epoch=current_epoch - 1)
        #     # self.LrUpdater = torch.optim.lr_scheduler.ReduceLROnPlateau(self.Optimizer)
        # self.LrUpdater.step()
        # """Sets the learning rate 手动调节
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
        raise NotImplemented

    @abstractmethod
    def get_dataloader(self, train_image_paths: list, train_image_labels: list, validate_image_paths: list, validate_image_labels: list, input_image_size: int, batch_size: int) -> tuple[TorchDataset, TorchDataset]:
        '''
        根据传入的图片路径和标签序列
        初始化数据集，从而确定图片预处理步骤，并初始化训练集和验证集
        '''
        assert len(train_image_paths) == len(train_image_labels)
        assert len(validate_image_paths) == len(validate_image_labels)
        # 默认情况下，使用默认的图片预处理步骤，如果需要拓展或者使用自定义的 loader，则在子类中重写本方法
        train_dataset = ClassifyTraining_Dataset(train_image_paths, train_image_labels, input_image_size)
        validate_dataset = ClassifyTraining_Dataset(validate_image_paths, validate_image_labels, input_image_size)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def _train_or_eval_one_epoch(self, current_epoch: int, is_train: bool) -> None:
        '''
        执行一次 epoch 的训练或测试
        '''
        write_to_session = 'Train' if is_train else 'Eval'
        loader = self.train_loader if is_train else self.eval_loader
        item_count = len(loader.dataset)
        batch_count = len(loader)
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
            if is_train:
                self.Model.train()
                # forward
                outputs = self.Model(images)
                tensor_loss = self.loss_func(outputs, labels)
                # backward
                self.Optimizer.zero_grad()
                tensor_loss.backward()
                self.Optimizer.step()
            else:
                self.Model.eval()
                with torch.no_grad():
                    outputs = self.Model(images)
                    tensor_loss = self.loss_func(outputs, labels)
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
        if is_train:
            # 更新 LR
            self.optimizer_lr_adjust(self.Settings.LR, current_epoch)

        # 计算平均值
        loss = loss / item_count
        acc_top1 = acc_top1 / item_count
        acc_top1_90 = acc_top1_90 / item_count
        acc_top5 = acc_top5 / item_count
        # 打印
        self.logger.info('epoch: %d, loss: %.4f, acc_top1: %.4f, acc_top1_90: %.4f, acc_top5: %.4f' % (current_epoch, loss, acc_top1, acc_top1_90, acc_top5))
        print('LR = ', self.Optimizer.param_groups[0]['lr'])
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
            if is_best or current_epoch % self.Settings.SaveModelEpoch == 0:
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
        self.loss_func = self.init_loss_function()
        if self.Settings.UseGpuCount > 0:
            for group in self.Optimizer.param_groups:
                group.setdefault('initial_lr', self.Settings.LR)
                group.setdefault('initial_lr', self.Settings.ResumeLR)

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

        # 设置训练集和验证集
        train_image_paths, train_image_labels, validate_image_paths, validate_image_labels = ClassifyTraining_Dataset.get_train_validate_image_list(self.label_info_csv_path, validate_ratio=0.2)

        # 开始训练
        for current_epoch in range(self.Settings.ResumeEpoch, self.Settings.Epochs):
            # 创建 loader
            self.train_loader, self.eval_loader = self.get_dataloader(train_image_paths=train_image_paths, train_image_labels=train_image_labels, validate_image_paths=validate_image_paths, validate_image_labels=validate_image_labels,
                                                                      input_image_size=self.Settings.InputSize,
                                                                      batch_size=self.Settings.BatchSize)
            self.logger.info("train_loader items count = {}, epoch count = {}".format(len(self.train_loader.dataset), len(self.train_loader)))
            self.logger.info("eval_loader items count = {}, epoch count = {}".format(len(self.eval_loader.dataset), len(self.eval_loader)))
            self.logger.info('epoch {} start'.format(current_epoch))
            self._train_or_eval_one_epoch(current_epoch, True)
            self._train_or_eval_one_epoch(current_epoch, False)
            pass

    def test(self, image_path: str, epoch: int = -1):
        '''
        测试一张图片，输入 epoch 小于 0 时，使用 the best 模型
        '''
        from ClassifyPreprocess_SingleImageAugmenter import SingleImageAugmenter
        labels = DatasetAnalyser.ReadFromCsv(self.Settings.DatasetLabelInfoCsvPath).labels
        labels = dict((item.label_value, item.label_name) for item in labels)
        from torchvision import transforms
        rbg = SingleImageAugmenter.open_image_by_opencv_as_rgb(image_path, self.Settings.InputSize)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        my_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        image = transforms.ToTensor()(rbg)
        image = transforms.Normalize(mean=mean, std=std)(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        model = self.load_checkpoint(epoch) if epoch >= 0 else self.load_best_pth()
        model.eval()
        with torch.no_grad():
            output = model(image)[0]
            print(output)
            # 输出 top5
            percentages, indices = torch.sort(output, descending=True)
            percentages = percentages.softmax(0)
            for j in range(5):
                print('{}({}) - {:.8f}'.format(labels[indices[j].item()], indices[j].item(), percentages[j]))
        pass


# if __name__ == '__main__':
#     # s = ClassifyTraining_Settings()
#     # s.set_dataset_path(r'D:\UritWorks\AI\image_preprocess\Augmented')
#     # if os.path.exists(s.OutputDirPath) == False:
#     #     os.makedirs(s.OutputDirPath)
#     # setting_path = os.path.join(s.OutputDirPath, 'demo_train_setting.json')
#     # s.to_json_file(setting_path)
#     # t = ITrainer(setting_path)
#     # t.train()
#     t = ITrainer(r'D:\UritWorks\AI\image_preprocess\DemoTrained\demo_traind_setting.json_20220311113655.json')
#     t.train()
