import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

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

def extract_list(list_path: str):
    with open(list_path) as f:
        list = f.read().splitlines()

    return list

def create_split(list, val_size = 0.2, seed = 42, shuffle = True):
    train_list, val_list = train_test_split(
        list, 
        test_size = val_size,
        random_state = seed,
        shuffle = True
    )

    return train_list, val_list

class ChestXRayDataset(Dataset):
    def __init__(self, img_dir: str, csv_path: str, list_index: list, transform=None):        
        self.img_dir = img_dir
        self.transform = transform #preprocessing pipeline

        #preparing pandas dataframe
        df = pd.read_csv(csv_path)

        #keeping only images in list_index
        if list_index:
            df = df[df['Image Index'].isin(list_index)]

        self.df = df.reset_index(drop=True)        
        

    def __getitem__(self, idx):
        row = self.df.iloc[idx] #extracting the df row

        #opening and preprocessing image
        image = Image.open(os.path.join(self.img_dir, row['Image Index'])).convert("L") #just the luminosity channel (gray-scale)
        image = image.convert("RGB") #ResNet wants 3 input channels

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

