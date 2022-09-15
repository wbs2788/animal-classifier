import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

IMAGENET_DEFAULT_MEAN = (0.5019333, 0.49997845, 0.44068748)
IMAGENET_DEFAULT_STD = (0.28612566, 0.26815864, 0.28786656)

def build_transform(is_train=False, input_size=224, crop_pct=None):
    resize_im = input_size > 32

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if input_size >= 384:  
            t.append(
            transforms.Resize((input_size, input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {input_size} size input images...")
        else:
            if crop_pct is None:
                crop_pct = 224 / 256
            size = int(input_size / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)