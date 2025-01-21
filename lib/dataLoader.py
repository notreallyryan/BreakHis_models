from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image
import os

def make_dataloader(path, batch_size = 32, train = False):
    """
    Specific Function to create a dataloader object linked to a given path and apply tranformations
    Takes the following inputs:
    - path: the folder path in which data is stored. For binary data, should contain two subfolders: 0 and 1.
    - batch_size: training_batch size. Defaults to 32
    - Train: boolean determining if dataloader is to be used for training or validation. If True, applies the transformations to prevent overfitting
    
    """

    #defining transformations, but only for training
    if train is True:
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale = (0.8, 1.0)),
            transforms.ToTensor()
        ])
    else: 
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    data = datasets.ImageFolder(root = path, transform = t)
    loader = DataLoader(data, batch_size, shuffle = True)
    return loader


    