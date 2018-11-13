import os
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms as T
import glob
from config import *
import random
random.seed(128)

class Data(data.Dataset):

    def __init__(self, data_path, mode):
        img_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
        if mode == "train" or mode == "val":
            random.shuffle(img_paths)
        split = int(len(img_paths)*0.75)
        if mode == "train":
            self.img_paths = img_paths[:split]
        elif mode == "val":
            self.img_paths = img_paths[split:]
        else:
            self.img_paths = img_paths

        if mode == "train" or mode == "val":
            self.labels = [path.split("/")[-1].split(".")[0] for path in self.img_paths]
            self.labels = [int(label == "cat") for label in self.labels]
        else:
            self.img_ids = [path.split("/")[-1].split(".")[0] for path in self.img_paths]

        self.transforms = T.Compose([
            T.Grayscale(),
            T.Resize((RE_HEIGHT, RE_WIDTH)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        self.mode = mode

    def __getitem__(self, index):

        img = Image.open(self.img_paths[index])
        img = self.transforms(img)

        if self.mode == "train" or self.mode == "val":
            label = self.labels[index]
            return img, label
        else:
            id = self.img_ids[index]
            return img, id

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    Data("./data/train")

