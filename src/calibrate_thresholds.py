from eval import validate 
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import ChestXRayDataset, extract_list, create_split, CLASSES
from preprocess import test_transform
from model import ResNet50
from torch.utils.data import DataLoader
import argparse 
import torch
import os
import numpy as np
import json
from pathlib import Path

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#parsing from CLI data root folder
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type = str,
    required = True,
    help = "Path to main dataset folder"
)
parser.add_argument(
    "--checkpoint",
    type = str,
    required = True,
    help = "Path to checkpoint model to use"
)
args = parser.parse_args()

#saving file paths
train_val_name = "train_val_list.txt"
csv_name = "Data_Entry_2017.csv" #hard-coded because the code refers to Kaggle dataset structure
csv_path = os.path.join(args.data_root, csv_name)
train_val_list_path = os.path.join(args.data_root, train_val_name)

#working with lists
train_val_list = extract_list(train_val_list_path)
train_list, val_list = create_split(train_val_list) #using default option, we only need validation set

#re-creating validation dataset and dataloader
val_data = ChestXRayDataset(data_root = args.data_root, csv_path = csv_path, list_index = val_list, transform = test_transform())
val_dl = DataLoader(val_data, batch_size = 64, num_workers = 4, pin_memory = True, persistent_workers = True, shuffle = False)

#importing best saved model
model = ResNet50()
model.load_state_dict(torch.load(args.checkpoint, map_location = device, weights_only = False)['model'])
model = model.to(device)


#getting all probs and labels for validation set
all_probs, all_labels, _  = validate(model, val_dl, device)

#to get best individual threshold for each class that maximize f1 score
def get_best_f1_threshold(all_probs, all_labels, step = 0.01):
    best_thresholds = np.zeros(all_probs.shape[1])
    best_stats = []
    thresholds = np.arange(step, 1, step)

    for c in range(all_probs.shape[1]):
        p = all_probs[:, c]
        y_true = all_labels[:, c].astype(int)

        best_f1, best_thr = -1.0, 0.5 #0.5 is the default threshold choice
        best_prec, best_rec = 0.0, 0.0

        for thr in thresholds:
            #calculating predictions based on output p + current threshold
            y_pred = (p >= thr).astype(int) 

            if np.sum(y_pred) == 0: #we want to avoid a threshold if there's no positive predictions (too aggressive & f1 = 0)
                continue

            #select threshold and save metrics if f1 is maximum
            f1 = f1_score(y_true, y_pred, zero_division = 0)
            if f1 > best_f1:
                best_thr = thr
                best_f1 = f1
                best_prec = precision_score(y_true, y_pred, zero_division = 0)
                best_rec = recall_score(y_true, y_pred, zero_division = 0)

        best_thresholds[c] = best_thr
        best_stats.append([best_f1, best_prec, best_rec])

    return best_thresholds, best_stats


def dict_stats(best_stats, classes: list):
    stats_per_class = {}
    
    for i, c in enumerate(classes):
        c_dict = {
            "F1": best_stats[i][0],
            "Precision": best_stats[i][1],
            "Recall": best_stats[i][2]
        }

        stats_per_class[c] = c_dict

    return stats_per_class


def print_save_calibration(best_thresholds, best_stats, out_path: str, method = "F1 maximization", split = "validation"):
    #printing results
    print(f"Best tresholds according to F1 score: {best_thresholds}")
    print(f"F1 scores: {best_stats[:, 0]}")
    print(f"Precision scores: {best_stats[:, 1]}")
    print(f"Recall score: {best_stats[:, 2]}") 

    #saving json file with thresholds
    data = {
        "metadata":{
            "method": method, 
            "split": split,
            "num_classes": len(CLASSES)
        },
        "thresholds":{
            c: float(thr) for c, thr in zip(CLASSES, best_thresholds)
        },
        "metrics": dict_stats(best_stats, CLASSES)
    }

    #creating parent dir if needed
    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)

    #opening and writing out file
    with open(out_path, "w") as f:
        json.dump(data, f, indent = 2)

    print(f"Thresholds saved to {out_path}.")


#extracting thresholds and metrics -> printing and saving
best_thresholds, best_stats = get_best_f1_threshold(all_probs, all_labels)
print_save_calibration(best_thresholds, best_stats, out_path="thresholds.json")