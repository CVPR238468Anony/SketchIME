#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logging import root
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from utils import get_mean_std_dataset
from torchvision.datasets.folder import *
from typing import *
from get_mean_std_dataset import *

class OfficeAmazonDataset(Dataset):
    """Class to create an iterable dataset
    of images and corresponding labels """

    def __init__(self, image_folder_dataset, transform=None):
        super(OfficeAmazonDataset, self).__init__()
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset.imgs)

    def __getitem__(self, idx):
        # read image, class from folder_dataset given index
        img, img_label = self.image_folder_dataset[idx][0], self.image_folder_dataset[idx][1]

        # apply transformations (it already returns them as torch tensors)
        if self.transform is not None:
            self.transform(img)

        img_label_pair = {"image": img,
                         "class": img_label}

        return img_label_pair


class SketchImageFolder(datasets.ImageFolder):
    """A sketch data loader where the dirnames are the label indexs
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root=root,
            #loader=loader,
            #IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            )
        

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name)-1 for i, cls_name in enumerate(classes)}

        return classes, class_to_idx


def get_dataloader(dataset, batch_size, train_ratio=0.7):
    """
    Splits a dataset into train and test.
    Returns train_loader and test_loader.
    """

    def get_subset(indices, start, end):
        return indices[start:start+end]

    # Split train/val data ratios
    TRAIN_RATIO, VALIDATION_RATIO = train_ratio, 1-train_ratio
    train_set_size = int(len(dataset) * TRAIN_RATIO)
    validation_set_size = int(len(dataset) * VALIDATION_RATIO)

    # Generate random indices for train and val sets
    indices = torch.randperm(len(dataset))
    train_indices = get_subset(indices, 0, train_set_size)
    validation_indices = get_subset(indices,train_set_size,validation_set_size)

    # Create sampler objects
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(validation_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size,sampler=val_sampler, num_workers=4)
    return train_loader, val_loader

def get_sketch_dataloader(name_dataset, batch_size, train=True):
    """
    Creates dataloader for the datasets in office datasetself.
    Uses get_mean_std_dataset() to compute mean and std along the
    color channels for the datasets in office.
    """

    # root dir (local pc or colab)
    root_dir = "../DataSet/%s" % name_dataset 
    
    # Ideally compute mean and std with get_mean_std_dataset.py
    mean_std = get_mean_std_dataset(root_dir) 

    # compose image transformations
    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std["mean"],std=mean_std["std"])
        ])

    # retrieve dataset using ImageFolder
    # datasets.ImageFolder() command expects our data to be organized
    # in the following way: root/label/picture.png
    # print("transform:",data_transforms)
    # print("type-transform:",type(data_transforms))
    dataset = SketchImageFolder(root=root_dir,transform=data_transforms)

    # Dataloader is able to spit out random samples of our data,
    # so our model won’t have to deal with the entire dataset every time.
    # shuffle data when training
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=train,num_workers=4,drop_last=True)

    return dataset_loader
    
    

