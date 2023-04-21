import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class FathomDataset(Dataset):
    def __init__(self, dataset, root, transform=None) -> None:
        super().__init__()
        self.data = dataset
        self.data['id'] = self.data['id'] + '.jpg'
        if root is not None:
            self.data['id'] = root + '/' + self.data['id']
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.data['id'][idx]).convert('RGB')
        target = torch.tensor(self.data['categories'][idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, target