from ast import Raise
import os
import numpy as np
import torchvision as tv
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile        
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 255.
        # img -= 1.0
        return img.transpose((2, 0, 1))

class BaseDataset(Dataset):
    def __init__(self, path_data):
        super(BaseDataset, self).__init__()
        self.path_data = path_data
        self.transform = tv.transforms.Compose([tv.transforms.RandomCrop(256), Preprocess()])
        self.imgs = self.get_imgs(path_data)
    
    def get_imgs(self, path_data):
        raise NotImplementedError("not Implemented")

    def __len__(self):
        return len(self.metas)
    
    def __getitem__(self, idx):
        path = self.imgs[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            print(path)
        w, h = img.size
        if w < 256 or h < 256:
            m = min(w, h)
            if m == w:
                img = img.resize((256, int(h * 256 / m)), Image.BILINEAR)
            else:
                img = img.resize((int(w * 256 / m), 256), Image.BILINEAR)
        img = self.transform(img)
        return img

class CLICImageDataset(BaseDataset):
    def __init__(self, path_data):
        super(CLICImageDataset, self).__init__(path_data)
        self.path_data = path_data
        self.metas = self.get_imgs(path_data)

    def get_imgs(self, path_data):
        imgs = glob.glob(os.path.join(path_data, '*.png'))
        return imgs


class VimeoDataset(BaseDataset):
    def __init__(self, path_data):
        super(VimeoDataset, self).__init__(path_data)
        self.path_data = path_data
        self.metas = self.get_imgs(path_data)

    def get_imgs(self, path_data):
        imgs = glob.glob(os.path.join(path_data, '*/*/*.png'))
        return imgs