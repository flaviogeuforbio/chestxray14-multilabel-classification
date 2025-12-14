import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path

#Classes of the dataset
CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

def extract_list(list_path: str): #extract List object from list file path
    with open(list_path) as f:
        list = f.read().splitlines()

    return list

def create_split(list, val_size = 0.2, seed = 42, shuffle = True): #train/validation list split
    train_list, val_list = train_test_split(
        list, 
        test_size = val_size,
        random_state = seed,
        shuffle = True
    )

    return train_list, val_list

def create_image_map(data_root): #creating image map for kaggle (original) dataset layout
    image_map = {}

    for folder in Path(data_root).glob("images_*"):
        if folder.is_dir():
            images_subdir = folder / "images"

            for img in images_subdir.glob("*.png"):
                image_map[img.name] = str(img)

    return image_map


class ChestXRayDataset(Dataset):
    def __init__(self, data_root: str, csv_path: str, list_index: list, transform=None):        
        self.data_root = data_root
        self.image_map = create_image_map(data_root)
        self.transform = transform #preprocessing pipeline

        #preparing pandas dataframe
        df = pd.read_csv(csv_path)

        #keeping only images in list_index
        if list_index:
            df = df[df['Image Index'].isin(list_index)]

        #removing non-existent entries
        available_images = set(self.image_map.keys())
        df = df[df['Image Index'].isin(available_images)] #deleting rows corr. to images actually not in the dataset

        self.df = df.reset_index(drop=True)        
        

    def __getitem__(self, idx):
        row = self.df.iloc[idx] #extracting the df row

        #opening and preprocessing image
        image_path = self.image_map[row['Image Index']]
        image = Image.open(image_path).convert("RGB") #ResNet wants RGB inputs

        if self.transform:
            image = self.transform(image) #applying preprocessing pipeline

        #creating the multi-hot encoded label
        label = torch.zeros(14) #14 = number of classes 
        diagnosis = row['Finding Labels'].split('|')
        for i, c in enumerate(CLASSES):
            if c in diagnosis:
                label[i] = 1.0
        
        return image, label


    def __len__(self):
        return len(self.df)

