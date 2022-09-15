import os

from PIL import Image
from torchvision.datasets import ImageFolder, VisionDataset

def read_image(path, shape=None, resample=None):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def read_images(path, shape=None, resample=None):
    imgs = []
    valid_exts = \
        set([".jpg", ".gif", ".png", 
            ".tga", ".jpeg", ".ppm"])
    for img_name in sorted(os.listdir(path)):
        ext = os.path.splitext(img_name)[-1]
        if ext.lower() not in valid_exts:
            continue
        img = read_image(os.path.join(path, img_name), 
                         shape, resample)
        if img == None:
            continue
        imgs.append(img)
    # imgs = np.stack(imgs, axis=0)

    return imgs

class InputData(VisionDataset):
    def __init__(self, root: str, transforms=None, 
                transform=None, 
                target_transform=None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.transform = transform
        self.data = read_images(root)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        
        return img, 0