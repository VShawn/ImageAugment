import os
import json
from ClassifyPreprocess_DataSetAnalyser import DataSetAnalyser, LabelInfo
from ClassifyPreprocess_SingleImageAugmenter import SingleImageAugmenter
class DataSetAugmenter(object):
    '''
    数据集扩充器
    '''
    def __init__(self, topPath):
        self.topPath = topPath
        self.org_dataset_analyzer = DataSetAnalyser(topPath)
        # 读取默认 SingleImageAugmenter 配置
        self.global_augment_settings = SingleImageAugmenter()
        if not os.path.exists(os.path.join(topPath, 'augment_settings.json')):
            # 如果 'augment_settings.json' 不存在，则创建一个默认配置
            self.global_augment_settings.ToJsonFile(os.path.join(topPath, 'augment_settings.json'))
        else:
            self.global_augment_settings = SingleImageAugmenter.FromJsonFile(os.path.join(topPath, 'augment_settings.json'))
        return

    # 扩充数据，并保存到文件夹
    def Run(self, saveDirPath: str):
        # 创建保存文件夹
        if not os.path.exists(saveDirPath):
            os.makedirs(saveDirPath)
        # 遍历标签 self.org_dataset_analyzer.labels
        for label in self.org_dataset_analyzer.labels:
            # 扩充标签
            self.__AugmentOneLabel(label, saveDirPath)
            pass
        return

    def __AugmentOneLabel(self, label: LabelInfo, saveDirPath: str):
        # 原始数据集文件夹不存在时抛出异常
        if not os.path.exists(label.label_path):
            raise Exception('{} is not exist!'.format(label.label_path))

        # 计算当前标签扩充后存储路径
        saveDirPath = os.path.join(saveDirPath, os.path.basename(label.label_path))
        if not os.path.exists(saveDirPath):
            os.makedirs(saveDirPath)

        # 读取当前标签扩充配置
        label_augmenter = self.global_augment_settings
        # 若存在 json 文件，则读取该类型定制的扩充配置
        if os.path.exists(os.path.join(label.label_path, 'augment_settings.json')):
            label_augmenter = SingleImageAugmenter.FromJsonFile(os.path.join(label.label_path, 'augment_settings.json'))
        # 计算单个图片需扩充数量
        augment_one_image_count = label_augmenter.AugmentCount / label.image_count
        # 创建单个图片扩充器
        img_augmenter = SingleImageAugmenter.FromJson(label_augmenter.ToJson()) # 拷贝一份
        img_augmenter.SetAugmentCount(augment_one_image_count) # 设置单个图片扩充数量
        # 开始扩充
        for image_path in label.image_paths:
            img_augmenter.RunByImagePathAndSave(image_path, saveDirPath)
            pass
        return



if __name__ == '__main__':
    # 分析指定文件夹下的文件夹名称，生成一个配置文件
    aug = DataSetAugmenter('E:\BM3000-TEST\B\FiveCells')
    aug.Run('Augmented')