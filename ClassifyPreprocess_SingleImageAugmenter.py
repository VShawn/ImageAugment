import os
import json
from enum import Enum
from enum import IntEnum
from itertools import count
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from numpy import ndarray

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
    '''
    def __init__(self) -> None:
        self.MakeBorderMode = AugmentMakeBorderMode.symmetric # 边缘填充方法 symmetric: 镜面对称填充； edge: 边缘填充；const：固定值填充
        self.MakeBorderConstValue = (125, 132) # 固定值填充时的值
        self.Step0_FlipCount = 3 # 翻转次数，0：不翻转，1：水平，2：水平 & 垂直，3：水平 & 垂直 & 水平垂直
        self.Step1_RotateCount = 100 # 旋转次数
        self.Step1_RotateMode = AugmentRotateMode.Average # 旋转模式，默认为平均旋转，即每次旋转 360° / (RotateCount + 1)
        self.Step2_ScaleCount = 2 # 缩放次数，设置为 0 时表示不缩放，只输出原始尺寸图像，否则输出图像为：原始尺寸图像 + ScaleCount 张缩放图像
        self.Step2_ScaleRange = (0.9, 1.1) # 缩放范围，在范围内随机选取 ScaleCount 个缩放系数执行缩放，默认为 (0.9, 1.1)
        self.Step3_ContrastAdjustCount = 0 # 对比度调整次数，设置为 0 时表示不调整对比度
        self.Step3_ContrastAdjust = (0.9, 1.1) # 对比度调整范围，在范围内随机选取 1 个对比度调整系数执行对比度调整，默认为 (0.9, 1.1)
        self.Step4_GaussianNoiseCount = 2 # 高斯噪声次数，设置为 0 时表示不添加高斯噪声
        self.Step4_GaussianNoiseRange = (0, 5) # 高斯噪声范围，在范围内随机选取 GaussianNoiseCount 个高斯噪声系数执行高斯噪声，默认为 (0, 5)

        self._step0_fliplr = ia.augmenters.Fliplr(1.0)
        self._step0_flipud = ia.augmenters.Flipud(1.0)
        pass


    def __json__(self):
        return {
            'MakeBorderMode': self.MakeBorderMode,
            'MakeBorderConstValue': self.MakeBorderConstValue,
            'Step0_FlipCount': self.Step0_FlipCount,
            'Step1_RotateCount': self.Step1_RotateCount,
            'Step1_RotateMode': self.Step1_RotateMode,
            'Step2_ScaleCount': self.Step2_ScaleCount,
            'Step2_ScaleRange': self.Step2_ScaleRange,
            'Step3_ContrastAdjustCount': self.Step3_ContrastAdjustCount,
            'Step3_ContrastAdjust': self.Step3_ContrastAdjust,
            'Step4_GaussianNoiseCount': self.Step4_GaussianNoiseCount,
            'Step4_GaussianNoiseRange': self.Step4_GaussianNoiseRange,
        }
        pass

    def ToJson(self) -> str:
        return json.dumps(self, default=lambda o: o.__json__(), indent=4, separators=(',', ': '))

    def ToJsonFile(self, filePath):
        with open(filePath, 'w') as f:
            f.write(self.ToJson())
        pass


    @staticmethod
    def FromJson(strJson: str) -> 'SingleImageAugmenter':
        jobj = json.loads(strJson)
        ret = SingleImageAugmenter()
        ret.MakeBorderMode = AugmentMakeBorderMode(jobj["MakeBorderMode"])
        ret.MakeBorderConstValue = jobj["MakeBorderConstValue"]
        ret.Step0_FlipCount = jobj["Step0_FlipCount"]
        ret.Step1_RotateCount = jobj["Step1_RotateCount"]
        ret.Step1_RotateMode = AugmentRotateMode(jobj["Step1_RotateMode"])
        ret.Step2_ScaleCount = jobj["Step2_ScaleCount"]
        ret.Step2_ScaleRange = jobj["Step2_ScaleRange"]
        ret.Step3_ContrastAdjustCount = jobj["Step3_ContrastAdjustCount"]
        ret.Step3_ContrastAdjust = jobj["Step3_ContrastAdjust"]
        ret.Step4_GaussianNoiseCount = jobj["Step4_GaussianNoiseCount"]
        ret.Step4_GaussianNoiseRange = jobj["Step4_GaussianNoiseRange"]
        return ret

    @staticmethod
    def FromJsonFile(strJson: str) -> 'SingleImageAugmenter':
        with open(strJson, 'r') as f:
            return SingleImageAugmenter.FromJson(f.read())

    def SetOutputCount(self, outCount: int) -> None:
        self.Step1_RotateCount = int(outCount / (1 + self.Step0_FlipCount) / (1 + self.Step2_ScaleCount) / (1 + self.Step3_ContrastAdjustCount) / (1 + self.Step4_GaussianNoiseCount))
        pass

    def RunByImagePath(self, imagePath: str) -> list:
        '''
        执行扩充
        :param imgPath: 图像路径
        :return: 扩充后的图像列表
        '''
        image = imageio.imread(imagePath) # 读取格式为 RGB # 若用 opencv 读取，则格式为 BGR，因此需要转换
        return self.RunByImageioImage(image)

    def RunByImageioImage(self, image: ndarray) -> list:
        '''
        执行扩充，输入为 imageio.imread(imagePath) 读取的图片 RGB 格式。
        :param image: 图像
        :return: 扩充后的图像列表
        '''
        # Step0: 翻转
        flipImgs = []
        flipImgs.append(image)
        if self.Step0_FlipCount == 1:
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
                else: # self.Step1_RotateMode == AugmentRotateMode.Random:
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
        contrastImgs =[]
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

    def RunByImagePathAndSave(self, imagePath: str, saveDirPath: str) -> None:
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
        ext = os.path.splitext(imagePath)[1]
        # 提取文件名，不包含扩展名
        fileName = os.path.splitext(os.path.basename(imagePath))[0]
        imgs = self.RunByImagePath(imagePath)
        for i, img in enumerate(imgs):
            savePath = os.path.join(saveDirPath, '{}_#{}{}'.format(fileName, i, ext))
            imageio.imwrite(savePath, img)
            pass

if __name__ == '__main__':
    aff = SingleImageAugmenter()
    aff.ToJsonFile('test.json')
    aff2 = SingleImageAugmenter.FromJsonFile('test.json')
    print(aff2.ToJson())

    aff2.SetOutputCount(1000)
    aff2.RunByImagePathAndSave('test.jpg', 'out')

    aff3 = SingleImageAugmenter()
    aff3.SetOutputCount(500)
    aff3.Step1_RotateMode = AugmentRotateMode.Random
    results = aff3.RunByImagePath('test_gray.jpg')
    os.makedirs('out_gray', exist_ok=True)
    for i in range(len(results)):
        imageio.imwrite('out_gray/test_gray_#' + str(i) + '.jpg', results[i])


