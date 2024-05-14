import PIL
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torch



class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        labels = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            img_path = words[0]
            label = int(words[1])  # Convert labels to integers

            try:
                img = Image.open(img_path).convert('RGB')
                imgs.append(img_path)
                labels.append(label)
            except PIL.UnidentifiedImageError:
                print(f"Skipping image: {img_path}")

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        # onehot编码
        label = F.one_hot(torch.tensor(label), num_classes=5).float()
        if self.target_transform is not None:
            label = self.target_transform(label)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
