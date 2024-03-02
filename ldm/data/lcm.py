import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class LCMBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        interpolation="bicubic",
        size = None
    ):
        self.prompt_path = txt_file
        self.data_root = data_root
        with open(self.prompt_path, "r") as f:
            self.prompts = f.read().splitlines()
        self._length = len(self.prompts)
        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        example  = {}
        example["caption"] = self.prompts[idx]
        img_name = f"{idx}.png"
        img_path = os.path.join(self.data_root, img_name)
        image = Image.open(img_path)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class LCMDatabaseTrain(LCMBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/ubuntu/dataset/lcm_prompts.txt", data_root="/home/ubuntu/dataset/lcm/", **kwargs)
    