import imageio
import imgaug as ia
import numpy as np
import cv2


bgr = cv2.imread("test.jpg")
image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# image = imageio.imread("test.jpg")
images = [image, image, image, image]

# print("Original:")
# ia.imshow(image)


from imgaug import augmenters as iaa
ia.seed(4)


# print("flip:")
# flipImgs = []
# flipImgs.append(image)
# augtmp = iaa.Fliplr(1.0)
# flipImgs.append(augtmp.augment_image(image))
# ia.imshow(np.hstack(flipImgs))


# rotate = iaa.Affine(
#     #rotate=(-25, 25)
#     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     # rotate=(-180, 180),
#     # scale=(0.9, 1.1),
#     shear={"x": (-15, 15), "y": (-15, 15),},
#     cval=(120,140),
#     mode="symmetric"
# )
# print("rotate:")
# # image_aug = rotate(image=image)
# images_aug = rotate(images=images)
# ia.imshow(np.hstack(images_aug))

# print("Crop")
# aug = iaa.Crop(percent=(0, 0.1), keep_size=False)
# # ia.imshow(np.hstack(images))
# for i in range(10):
#     ia.imshow(aug.augment_image(image))


seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1), keep_size=False),
    iaa.Resize({"height": 128, "width": "keep-aspect-ratio"}, ),
    iaa.PadToFixedSize(width=256, height=256, position="center", pad_mode="symmetric"),
])
ia.imshow(np.hstack(seq.augment_images(images)))


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