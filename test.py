import imageio
import imgaug as ia
import numpy as np

image = imageio.imread("1.jpg")

print("Original:")
ia.imshow(image)


from imgaug import augmenters as iaa
ia.seed(4)


flipImgs = []
flipImgs.append(image)
augtmp = iaa.Fliplr(1.0)
flipImgs.append(augtmp.augment_image(image))
ia.imshow(np.hstack(flipImgs))


rotate = iaa.Affine(
    #rotate=(-25, 25)
    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    # rotate=(-180, 180),
    # scale=(0.9, 1.1),
    shear={"x": (-15, 15), "y": (-15, 15),},
    cval=(120,140),
    mode="symmetric"
)
image_aug = rotate(image=image)

print("Augmented:")
ia.imshow(image_aug)


images = [image, image, image, image]
images_aug = rotate(images=images)

print("Augmented batch:")
ia.imshow(np.hstack(images_aug))


seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 5)),
    iaa.Affine(rotate=(0, 359),
        scale=(0.9, 1.1),
        cval=(120,140),
        mode="symmetric"),
    iaa.Crop(percent=(0, 0.1),keep_size=True)
])

mages_aug = seq(images=images)
ia.imshow(np.hstack(mages_aug))