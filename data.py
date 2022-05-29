from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader



def get_transforms():
    resize, cropsize = 512, 448

    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_eval

class IshiharaDataset(torch.utils.data.Dataset):
    def __init__(self, train_flag):
        self.train_flag = train_flag
        if self.train_flag:
            self.data_path = "data/ishihara/train"
        else:
            self.data_path = "data/ishihara/test"

        self.file_names = os.listdir(self.data_path)
        self.num_class = 10

        self.transform_train, self.transform_eval = get_transforms()

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.file_names[idx])
        #image = read_image(img_path)
        image = Image.open(img_path).convert('RGB')
        image = self.transform_train(image)
        label = int(self.file_names[idx].split('/')[-1][0])

        return image, label
    
    def __len__(self):
        return len(self.file_names)