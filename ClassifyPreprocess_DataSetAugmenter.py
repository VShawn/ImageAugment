from genericpath import exists
from importlib.resources import path
from pickle import NONE
import sys, getopt, os, json, re, time
import logging
from numpy import array
from ClassifyPreprocess_DataSetAnalyser import DataSetAnalyser, LabelInfo
from ClassifyPreprocess_SingleImageAugmenter import SingleImageAugmenter
class DataSetAugmenter(object):
    '''
    数据集扩充器
    '''
    def __init__(self, topPath, outDirPath, augment_settings_json_path: str = ""):
        '''
        构造函数
        :param topPath: 数据集根目录
        :param augment_setting: 输入的扩充配置文件路径，格式为 {'正则匹配串1' -> SingleImageAugmenter,'正则匹配串2' -> SingleImageAugmenter,}，
        若传入为空则会读取数据集根目录中的配置文件
        '''
        self.topPath = topPath
        self.outDirPath = outDirPath
        self.augment_settings_json_path = augment_settings_json_path

    def __basic_init(self):
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

        # 分析数据集，得到标签 self.org_dataset_analyzer.labels
        self.org_dataset_analyzer = DataSetAnalyser(self.topPath)

        self.logger.info('Start to augment dataset... {}'.format(self.topPath))
        self.logger.info('we have {} targets to augment'.format(len(self.org_dataset_analyzer.labels)))

        json_path:str = ""

        # 确定扩充配置文件路径
        if self.augment_settings_json_path != "" and os.path.exists(self.augment_settings_json_path):
            json_path = self.augment_settings_json_path
        # 使用数据集根目录中的配置文件路径
        else:
            json_path = os.path.join(self.topPath, 'augment_settings.json')
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
                    f.write(json.dumps({k: v.ToDict() for k, v in settings.items()}, indent=4, separators=(',', ': ')))
                print("we generated a default augment_settings.json in '{}'".format(json_path))

        logging.info('reading setting from: {}'.format(json_path))
        # 读取 json 配置文件
        tmp_dict = json.load(open(json_path, 'r'))
        self.augment_settings = {k:SingleImageAugmenter.FromDict(v) for k,v in tmp_dict.items()}

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
        self.org_dataset_analyzer.SaveToCsv(os.path.join(self.outDirPath, 'dataset_analysis.csv'))
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
        outDirPath = os.path.join(self.outDirPath, label.label_dir_name)
        if not os.path.exists(outDirPath):
            os.makedirs(outDirPath)

        # 遍历 self.augment_settings 找到匹配的配置，找不到就用 global_augment_setting
        label_augmenter : SingleImageAugmenter = None
        for key in self.augment_settings.keys():
            if re.search(key, label.label_dir_name, re.I) or re.search(key, label.label_name, re.I):
                label_augmenter = self.augment_settings[key]
        if label_augmenter == None:
            label_augmenter = self.global_augment_setting

        # 计算单个图片需扩充数量
        augment_one_image_count = label_augmenter.AugmentCount / label.image_count
        # 创建单个图片扩充器
        img_augmenter = SingleImageAugmenter.FromJson(label_augmenter.ToJson()) # 拷贝一份
        img_augmenter.SetAugmentCount(augment_one_image_count) # 设置单个图片扩充数量
        # 开始扩充
        for image_path in label.image_paths:
            img_augmenter.RunByImagePathAndSave(image_path, outDirPath)
            pass
        return



if __name__ == '__main__':
    argv = sys.argv[1:]
    inputDir = ''
    outputDir = ''
    settingsPath = ''
    # 读取输入参数
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["i=","o=","c="])
        if len(opts) == 0:
            # print('please set args: -i <input dir path> -o <output dir path>')
            # sys.exit()
            inputDir = 'E:\BM3000-TEST\B\FiveCells'
            outputDir = 'Augmented'
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <input dir path> -o <output dir path> -c <json augment_settings file path>')
                    sys.exit()
                else:
                    if not os.path.exists(arg):
                        raise '{} {} not existed'.format(opt, arg)
                    elif opt in ("-i"):
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
    aug = DataSetAugmenter(inputDir, outputDir)
    aug.Run()