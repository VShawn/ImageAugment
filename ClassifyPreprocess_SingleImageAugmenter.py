from math import floor
import os
import json
from enum import Enum
from enum import IntEnum
import cv2
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap


class AugmentMakeBorderMode(IntEnum):
    const = 0,
    edge = 1,
    symmetric = 2,


class AugmentRotateMode(IntEnum):
    Random = 1,
    Average = 2,


class SingleImageAugmenter(object):
    '''
    单个图像扩充器，输出图像数量为 (1 + Step0_FlipCount) * (1 + RotateCount) * (1 + ScaleCount) * (1 + GaussianNoiseCount)
    新建实例时先配置除了 Step1_RotateCount 外的参数，再 set_augment_count 设置扩充目标数量，自动计算旋转次数
    从 json 读取时，扩充目标数量已设置好了，不需要单独调用 set_augment_count 。
    '''

    def __init__(self) -> None:
        self.MakeBorderMode = AugmentMakeBorderMode.symmetric  # 边缘填充方法 symmetric: 镜面对称填充； edge: 边缘填充；const：固定值填充
        self.MakeBorderConstValue = (125, 132)  # 固定值填充时的值
        self.Step0_FlipCount = 3  # 翻转次数，0：不翻转，1：水平，2：水平 & 垂直，3：水平 & 垂直 & 水平垂直
        self.Step1_RotateCount = 100  # 旋转次数
        self.Step1_RotateMode = AugmentRotateMode.Average  # 旋转模式，默认为平均旋转，即每次旋转 360° / (RotateCount + 1)
        self.Step2_ScaleCount = 2  # 缩放次数，设置为 0 时表示不缩放，只输出原始尺寸图像，否则输出图像为：原始尺寸图像 + ScaleCount 张缩放图像
        self.Step2_ScaleRange = (0.9, 1.1)  # 缩放范围，在范围内随机选取 ScaleCount 个缩放系数执行缩放，默认为 (0.9, 1.1)
        self.Step3_ContrastAdjustCount = 0  # 对比度调整次数，设置为 0 时表示不调整对比度
        self.Step3_ContrastAdjust = (0.9, 1.1)  # 对比度调整范围，在范围内随机选取 1 个对比度调整系数执行对比度调整，默认为 (0.9, 1.1)
        self.Step4_GaussianNoiseCount = 2  # 高斯噪声次数，设置为 0 时表示不添加高斯噪声
        self.Step4_GaussianNoiseRange = (0, 5)  # 高斯噪声范围，在范围内随机选取 GaussianNoiseCount 个高斯噪声系数执行高斯噪声，默认为 (0, 5)
        self.set_augment_count(5000)  # 设置扩充目标数量，并根据扩充目标数量计算旋转次数
        self._step0_fliplr = ia.augmenters.Fliplr(1.0)
        self._step0_flipud = ia.augmenters.Flipud(1.0)
        pass

    def to_dict(self):
        return {
            'MakeBorderMode': self.MakeBorderMode,
            'MakeBorderConstValue': self.MakeBorderConstValue,
            'Step0_FlipCount': self.Step0_FlipCount,
            # 'Step1_RotateCount': self.Step1_RotateCount,
            'Step1_RotateMode': self.Step1_RotateMode,
            'Step2_ScaleCount': self.Step2_ScaleCount,
            'Step2_ScaleRange': self.Step2_ScaleRange,
            'Step3_ContrastAdjustCount': self.Step3_ContrastAdjustCount,
            'Step3_ContrastAdjust': self.Step3_ContrastAdjust,
            'Step4_GaussianNoiseCount': self.Step4_GaussianNoiseCount,
            'Step4_GaussianNoiseRange': self.Step4_GaussianNoiseRange,
            'AugmentCount': self.AugmentCount,
        }
        pass

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.to_dict(), indent=4, separators=(',', ': '))

    def to_json_file(self, filePath):
        with open(filePath, 'w') as f:
            f.write(self.to_json())
        pass

    @staticmethod
    def from_dict(dict_obj: dict) -> 'SingleImageAugmenter':
        ret = SingleImageAugmenter()
        ret.AugmentCount = dict_obj["AugmentCount"]
        ret.MakeBorderMode = AugmentMakeBorderMode(dict_obj["MakeBorderMode"])
        ret.MakeBorderConstValue = dict_obj["MakeBorderConstValue"]
        ret.Step0_FlipCount = dict_obj["Step0_FlipCount"]
        # ret.Step1_RotateCount = dict_obj["Step1_RotateCount"]
        ret.Step1_RotateMode = AugmentRotateMode(dict_obj["Step1_RotateMode"])
        ret.Step2_ScaleCount = dict_obj["Step2_ScaleCount"]
        ret.Step2_ScaleRange = dict_obj["Step2_ScaleRange"]
        ret.Step3_ContrastAdjustCount = dict_obj["Step3_ContrastAdjustCount"]
        ret.Step3_ContrastAdjust = dict_obj["Step3_ContrastAdjust"]
        ret.Step4_GaussianNoiseCount = dict_obj["Step4_GaussianNoiseCount"]
        ret.Step4_GaussianNoiseRange = dict_obj["Step4_GaussianNoiseRange"]
        ret.set_augment_count(ret.AugmentCount)
        return ret

    @staticmethod
    def from_json(strJson: str) -> 'SingleImageAugmenter':
        jobj = json.loads(strJson)
        return SingleImageAugmenter.from_dict(jobj)

    @staticmethod
    def from_json_file(strJson: str) -> 'SingleImageAugmenter':
        with open(strJson, 'r') as f:
            return SingleImageAugmenter.from_json(f.read())

    def set_augment_count(self, augmentCount: int) -> None:
        '''
        配置好其他参数后，根据 augmentCount 设置旋转次数
        :augmentCount: 扩充出的图片数量，输出数量必然大于或等于这个值
        '''
        self.AugmentCount = floor(augmentCount)
        if self.AugmentCount < 1:
            self.Step0_FlipCount = 0
            self.Step1_RotateCount = 0
            self.Step2_ScaleCount = 0
            self.Step3_ContrastAdjustCount = 0
            self.Step4_GaussianNoiseCount = 0
            return  # 如果扩充数量小于 1，则不进行任何扩充

        self.Step1_RotateCount = int(augmentCount / (1 + self.Step0_FlipCount) / (1 + self.Step2_ScaleCount) / (1 + self.Step3_ContrastAdjustCount) / (1 + self.Step4_GaussianNoiseCount))
        # 旋转次数超过 360 次，则增加 Step2_ScaleCount
        while self.Step1_RotateCount > 350:
            self.Step2_ScaleCount += 1
            self.Step1_RotateCount = int(self.AugmentCount / (1 + self.Step0_FlipCount) / (1 + self.Step2_ScaleCount) / (1 + self.Step3_ContrastAdjustCount) / (1 + self.Step4_GaussianNoiseCount))
        # 旋转次数为 0 则减少其他操作次数
        while self.Step1_RotateCount < 5 and self.Step1_RotateCount < self.AugmentCount:
            if self.Step0_FlipCount > 0:
                self.Step0_FlipCount -= 1
            elif self.Step2_ScaleCount > 0:
                self.Step2_ScaleCount -= 1
            elif self.Step3_ContrastAdjustCount > 0:
                self.Step3_ContrastAdjustCount -= 1
            elif self.Step4_GaussianNoiseCount > 0:
                self.Step4_GaussianNoiseCount -= 1
            self.Step1_RotateCount = int(self.AugmentCount / (1 + self.Step0_FlipCount) / (1 + self.Step2_ScaleCount) / (1 + self.Step3_ContrastAdjustCount) / (1 + self.Step4_GaussianNoiseCount))

    def run_by_image_path(self, image_path: str) -> list:
        '''
        执行扩充
        :param imgPath: 图像路径
        :return: 扩充后的图像列表
        '''
        # image = imageio.imread(image_path)  # 读取格式为 RGB # 若用 opencv 读取，则格式为 BGR，因此需要转换
        image: np.ndarray = self.open_image(image_path)
        return self.run_by_image_data(image)

    def run_by_image_data(self, image: np.ndarray) -> list:
        '''
        执行扩充，输入为 imageio.imread(image_path) 读取的图片 RGB 格式。
        :param image: 图像
        :augmentCount: 扩充出的图片数量，输出数量必然大于或等于这个值
        :return: 扩充后的图像列表
        '''
        # Step0: 翻转
        flipImgs = []
        flipImgs.append(image)
        if self.Step0_FlipCount == 0:
            pass
        elif self.Step0_FlipCount == 1:
            flipImgs.append(self._step0_fliplr.augment_image(image))
        elif self.Step0_FlipCount == 2:
            flipImgs.append(self._step0_fliplr.augment_image(image))
            flipImgs.append(self._step0_flipud.augment_image(image))
        elif self.Step0_FlipCount == 3:
            lr = self._step0_fliplr.augment_image(image)
            flipImgs.append(lr)
            flipImgs.append(self._step0_flipud.augment_image(image))
            flipImgs.append(self._step0_flipud.augment_image(lr))
        else:
            raise Exception('Step0_FlipCount 参数错误')

        # Step1: 旋转
        rotateImgs = []
        if self.Step1_RotateCount == 0:
            rotateImgs = flipImgs[:]
            pass
        else:
            for img in flipImgs:
                rotateImgs.append(img)
                if self.Step1_RotateMode == AugmentRotateMode.Average:
                    r = img
                    step = 360 / (self.Step1_RotateCount + 1)
                    rotater = iaa.Affine(rotate=step, mode=self.MakeBorderMode.name, cval=self.MakeBorderConstValue)
                    for i in range(self.Step1_RotateCount):
                        r = rotater.augment_image(r)
                        rotateImgs.append(r)
                else:  # self.Step1_RotateMode == AugmentRotateMode.Random:
                    rotater = iaa.Affine(rotate=(-180, 180), mode=self.MakeBorderMode.name, cval=self.MakeBorderConstValue)
                    for i in range(self.Step1_RotateCount):
                        rotateImgs.append(rotater.augment_image(img))

        # Step2: 缩放
        scaleImgs = []
        if self.Step2_ScaleCount == 0:
            scaleImgs = rotateImgs[:]
        else:
            scaler = iaa.Affine(scale=self.Step2_ScaleRange, mode=self.MakeBorderMode.name, cval=self.MakeBorderConstValue)
            for img in rotateImgs:
                scaleImgs.append(img)
                for i in range(self.Step2_ScaleCount):
                    scaleImgs.append(scaler.augment_image(img))

        # Step3: 对比度调整
        contrastImgs = []
        if self.Step3_ContrastAdjustCount == 0:
            contrastImgs = scaleImgs[:]
            pass
        else:
            contraster = ia.ContrastNormalization(self.Step3_ContrastAdjust)
            for img in scaleImgs:
                contrastImgs.append(img)
                for i in range(self.Step3_ContrastAdjustCount):
                    contrastImgs.append(contraster.augment_image(img))

        # Step4: 高斯噪声
        noiseImgs = []
        if self.Step4_GaussianNoiseCount == 0:
            noiseImgs = contrastImgs[:]
            pass
        else:
            noise = iaa.AdditiveGaussianNoise(scale=self.Step4_GaussianNoiseRange)
            for img in contrastImgs:
                noiseImgs.append(img)
                for i in range(self.Step4_GaussianNoiseCount):
                    noiseImgs.append(noise.augment_image(img))

        return noiseImgs

    def run_by_image_path_and_save(self, image_path: str, saveDirPath: str) -> None:
        '''
        执行扩充并保存
        :param image: 图像
        :param saveDirPath: 保存文件夹路径
        :return: None
        '''
        # 文件夹不存在则创建
        if not os.path.exists(saveDirPath):
            os.makedirs(saveDirPath)
        # 提取文件扩展名
        ext = os.path.splitext(image_path)[1]
        # 提取文件名，不包含扩展名
        fileName = os.path.splitext(os.path.basename(image_path))[0]
        imgs = self.run_by_image_path(image_path)
        for i, img in enumerate(imgs):
            savePath = os.path.join(saveDirPath, '{}_#{}{}'.format(fileName, i, ext))
            imageio.imwrite(savePath, img)
            pass

    @staticmethod
    def open_image(path: str, resize_to: int = None, random_crop_rate: tuple[float, float] = None) -> np.ndarray:
        """
        使用 opencv 打开图片
        path: 图片路径
        resize_to: 输出图片的尺寸，为 None 时直接输出原始尺寸，当图像大于 resize_to 时压缩图像，小于 resize_to 时以镜像方式扩充边缘
        random_crop_rate: 边缘随机裁切比例， 输入(0, 0.1) 表示随机裁切掉 0 - 10% 的边缘尺寸
        """
        bgr = cv2.imread(path)
        # BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if random_crop_rate is not None:
            # 随机裁切
            aug = iaa.Crop(percent=random_crop_rate, keep_size=False)
            rgb = aug.augment_image(rgb)
        if resize_to is not None:
            # 尺寸大于目标尺寸时，保持宽高比缩小
            h, w, _ = rgb.shape
            if h > resize_to or w > resize_to:
                if h > w:
                    h = resize_to
                    w = int(resize_to * w / h)
                else:
                    w = resize_to
                    h = int(resize_to * h / w)
                rgb = cv2.resize(rgb, (w, h))
            # 尺寸小于目标尺寸时，PAD
            h, w, _ = rgb.shape
            if h < resize_to or w < resize_to:
                pad = iaa.PadToFixedSize(width=resize_to, height=resize_to, position="center", pad_mode="symmetric")
                rgb = pad.augment_image(rgb)
            # 最终尺寸检查
            h, w, _ = rgb.shape
            if h != resize_to or w != resize_to:
                rgb = cv2.resize(rgb, (resize_to, resize_to))
        return rgb


if __name__ == '__main__':
    images = []
    ia.imshow(SingleImageAugmenter.open_image('test.jpg'))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    images.append(SingleImageAugmenter.open_image('test.jpg', resize_to=256, random_crop_rate=(0.0, 0.5)))
    ia.imshow(np.hstack(images))

    aff = SingleImageAugmenter()
    aff.to_json_file('test.json')
    aff2 = SingleImageAugmenter.from_json_file('test.json')
    print(aff2.to_json())
    aff2.set_augment_count(100)
    aff2.run_by_image_path_and_save('test.jpg', 'out')

    aff3 = SingleImageAugmenter()
    aff3.Step1_RotateMode = AugmentRotateMode.Random
    aff3.Step0_FlipCount = 0
    aff3.Step1_RotateCount = 0
    aff3.Step3_ContrastAdjustCount = 0
    aff3.Step4_GaussianNoiseCount = 0
    aff3.set_augment_count(50)
    results = aff3.run_by_image_path('test_gray.jpg')
    os.makedirs('out_gray', exist_ok=True)
    for i in range(len(results)):
        imageio.imwrite('out_gray/test_gray_#' + str(i) + '.jpg', results[i])
