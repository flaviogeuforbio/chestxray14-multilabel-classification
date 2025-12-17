from eval import validate
from dataset import ChestXRayDataset, extract_list, CLASSES
from preprocess import test_transform
from model import ResNet50
from torch.utils.data import DataLoader
import torch
import argparse
import os

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type = str,
    required = True,
    help = "Path to data root folder"
)
parser.add_argument(
    "--checkpoint",
    type = str,
    required = True, 
    help = "Path to checkpoint model to validate"
)
parser.add_argument(
    "--per_class",
    action = "store_true",
    help = "If True AUC score per class is returned, otherwise average AUC score"
)
args = parser.parse_args()
data_root = args.data_root
csv_name = "Data_Entry_2017.csv" #hand-coded because the code refers to Kaggle dataset structure
csv_path = os.path.join(data_root, csv_name)
list_path = os.path.join(data_root, "test_list.txt")

list_index = extract_list(list_path)

#Creating Dataset and DataLoader
test_data = ChestXRayDataset(
    data_root = data_root,
    csv_path = csv_path,
    list_index = list_index,
    transform = test_transform()
)

test_dl = DataLoader(test_data, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True, persistent_workers = True) 

#Loading the model
model = ResNet50()
model.load_state_dict(torch.load(args.checkpoint, map_location = device)['model'])
model = model.to(device)

#Calculating and showing AUC scores 
if args.per_class: #it shows AUC per each single class -> we can detect poorly managed classes
    auc_scores = validate(model, test_dl, device, average = False)
    class_scores = {c:s for (c,s) in zip(CLASSES, auc_scores)} #creating a readable dictionary {class: auc score}

    print(class_scores)

else:
    auc_score = validate(model, test_dl, device, average = True) #average AUC over classes
    
    print(auc_score)

