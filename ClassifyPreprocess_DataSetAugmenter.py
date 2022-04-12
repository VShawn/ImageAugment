import sys
import getopt
import os
import json
import re
import time
import logging
from math import floor
from ClassifyPreprocess_DataSetAnalyser import DatasetAnalyser, LabelInfo
from ClassifyPreprocess_SingleImageAugmenter import SingleImageAugmenter


class DatasetAugmenter(object):
    '''
    数据集扩充器
    '''

    def __init__(self, org_dataset_dir_path, augmented_dataset_dir_path, augment_settings_json_path: str = ""):
        '''
        构造函数
        :param org_dataset_dir_path: 数据集根目录
        :param augmented_dataset_dir_path: 扩充后的数据集保存路径
        :param augment_setting: 输入的扩充配置文件路径，格式为 {'正则匹配串1' -> SingleImageAugmenter,'正则匹配串2' -> SingleImageAugmenter,}，
        若传入为空则会读取数据集根目录中的配置文件
        '''
        self.org_dataset_dir_path = org_dataset_dir_path
        self.augmented_dataset_dir_path = augmented_dataset_dir_path
        self.augment_settings_json_path = augment_settings_json_path
        if not os.path.isabs(self.org_dataset_dir_path):
            self.org_dataset_dir_path = os.path.abspath(self.org_dataset_dir_path)
        if not os.path.isabs(self.augmented_dataset_dir_path):
            self.augmented_dataset_dir_path = os.path.abspath(self.augmented_dataset_dir_path)

    def __basic_init(self):
        # 创建保存文件夹
        if not os.path.exists(self.augmented_dataset_dir_path):
            os.makedirs(self.augmented_dataset_dir_path)

        # 日志写到当前时间的文件中
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        # 使用FileHandler输出到文件
        fh = logging.FileHandler(os.path.join(self.augmented_dataset_dir_path, '{}_{}.log'.format(time.strftime('%Y%m%d_%H%M%S'), os.path.basename(__file__))))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        # 添加两个Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 分析数据集，得到标签 self.org_dataset_analyzer.labels
        self.logger.info('Start to augment dataset... {}'.format(self.org_dataset_dir_path))
        self.org_dataset_analyzer = DatasetAnalyser()
        self.org_dataset_analyzer.SetPath(self.org_dataset_dir_path)
        self.logger.info('we have {} targets to augment'.format(len(self.org_dataset_analyzer.labels)))

        json_path: str = ""
        # 确定扩充配置文件路径
        if self.augment_settings_json_path != "":
            json_path = self.augment_settings_json_path
        # 使用数据集根目录中的配置文件路径
        else:
            json_path = os.path.join(self.org_dataset_dir_path, 'augment_settings.json')
        # 生成配置文件并退出
        if not os.path.exists(json_path):
            # 如果 'augment_setting.json' 不存在，则创建一个默认配置
            augment_setting = SingleImageAugmenter()
            settings = dict()
            settings['global'] = augment_setting
            for label in self.org_dataset_analyzer.labels:
                settings[label.label_dir_name] = augment_setting
                pass
            # 保存 dict 到 json
            with open(json_path, 'w') as f:
                f.write(json.dumps({k: v.to_dict() for k, v in settings.items()}, indent=4, separators=(',', ': ')))
            print("we generated a default augment_settings.json in '{}'".format(json_path))
            print("please modify it then rerun the script.'{}'".format(json_path))
            return

        logging.info('reading setting from: {}'.format(json_path))
        # 读取 json 配置文件
        tmp_dict = json.load(open(json_path, 'r'))
        self.augment_settings = {k: SingleImageAugmenter.from_dict(v) for k, v in tmp_dict.items()}

        # 检查 self.augment_settings 中是否有 'global'
        if 'global' in self.augment_settings:
            self.global_augment_setting = self.augment_settings['global']
            self.augment_settings.pop('global')
        else:
            self.global_augment_setting = SingleImageAugmenter()
        return

    # 扩充数据，并保存到文件夹
    def Run(self):
        self.__basic_init()
        # 保存分析结果到扩充文件夹
        # 创建输出信息
        augmented_dataset_analyzer = DatasetAnalyser()
        augmented_dataset_analyzer.SetPath(self.org_dataset_dir_path)
        for i in range(len(augmented_dataset_analyzer.labels)):
            augmented_dataset_analyzer.labels[i].label_dir_path = os.path.join(self.augmented_dataset_dir_path, os.path.basename(augmented_dataset_analyzer.labels[i].label_dir_path))
        augmented_dataset_analyzer.SaveToCsv(os.path.join(self.augmented_dataset_dir_path, 'label_info.csv'))
        # 遍历标签 self.org_dataset_analyzer.labels
        for i, label in enumerate(self.org_dataset_analyzer.labels):
            self.logger.info('start to augment label: {}, {}/{}'.format(label.label_dir_name, i + 1, len(self.org_dataset_analyzer.labels)))
            # 扩充标签
            self.__AugmentOneLabel(label)
            self.logger.info('finish augment label: {}, {}/{}'.format(label.label_dir_name, i + 1, len(self.org_dataset_analyzer.labels)))
            pass
        return

    def __AugmentOneLabel(self, label: LabelInfo):
        # 原始数据集文件夹不存在时抛出异常
        if not os.path.exists(label.label_dir_path):
            raise Exception('{} is not exist!'.format(label.label_dir_path))

        # 计算当前标签扩充后存储路径
        outDirPath = os.path.join(self.augmented_dataset_dir_path, label.label_dir_name)
        if not os.path.exists(outDirPath):
            os.makedirs(outDirPath)

        # 遍历 self.augment_settings 找到匹配的配置，找不到就用 global_augment_setting
        label_augmenter: SingleImageAugmenter = None
        for key in self.augment_settings.keys():
            if re.search(key, label.label_dir_name, re.I) or re.search(key, label.label_name, re.I):
                label_augmenter = self.augment_settings[key]
        if label_augmenter == None:
            label_augmenter = self.global_augment_setting

        # 计算单个图片需扩充数量
        augment_one_image_count: int = floor(label_augmenter.AugmentCount / label.image_count)
        # 创建单个图片扩充器
        img_augmenter = SingleImageAugmenter.from_json(label_augmenter.to_json())  # 拷贝一份
        img_augmenter.set_augment_count(augment_one_image_count)  # 设置单个图片扩充数量
        # 开始扩充
        for image_path in label.image_paths:
            img_augmenter.run_by_image_path_and_save(image_path, outDirPath)
            pass
        return


if __name__ == '__main__':
    argv = sys.argv[1:]
    inputDir = ''
    outputDir = ''
    settingsPath = ''
    generate_default_setting_path = ''
    # 读取输入参数
    try:
        opts, args = getopt.getopt(argv, "-h-i:-o:-c:", ["i=", "o=", "c="])
        if len(opts) == 0:
            print('please set args: -i <input dir path> -o <output dir path>')
            sys.exit()
            # inputDir = 'E:\BM3000-TEST\B\FiveCells'
            # outputDir = 'Augmented'
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <input dir path> -o <output dir path> -c <json augment_settings file path>')
                    sys.exit()
                else:
                    if opt in ("-i"):
                        inputDir = arg
                    elif opt in ("-o"):
                        outputDir = arg
                    elif opt in ("-c"):
                        settingsPath = arg
    except getopt.GetoptError:
        print('please set args: -i <input dir path> -o <output dir path> -c <json augment_settings file path>')
        sys.exit(2)

    print('输入为：', inputDir)
    print('输出为：', outputDir)
    print('配置文件', settingsPath)
    aug = DatasetAugmenter(inputDir, outputDir, settingsPath)
    aug.Run()
