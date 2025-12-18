from eval import validate
from dataset import ChestXRayDataset, extract_list, CLASSES
from preprocess import test_transform
from model import ResNet50
from torch.utils.data import DataLoader
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

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
csv_name = "Data_Entry_2017.csv" #hard-coded because the code refers to Kaggle dataset structure
csv_path = os.path.join(data_root, csv_name)
list_path = os.path.join(data_root, "test_list.txt")

# class_to_idx = {c:i for i, c in enumerate(CLASSES)} #map class name to idx ('Hernia': 13)

@torch.no_grad()
def get_class_dist(all_probs, all_labels, target_class):

    #extracting distribution for target class
    all_class_probs = all_probs[:, target_class]
    all_class_labels = all_labels[:, target_class]

    #getting output prob for positive & negative samples
    pos_samples = all_class_probs[all_class_labels == 1]
    neg_samples = all_class_probs[all_class_labels == 0]

    return pos_samples, neg_samples


def plot_class_dist(pos_samples, neg_samples, out_path: str):
    #plotting results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axs[0].violinplot([pos_samples, neg_samples],
                    showmeans=False,
                    showmedians=True)
    axs[0].set_title('Violin plot')

    #plot box plot
    axs[1].boxplot([pos_samples, neg_samples])
    axs[1].set_title('Box plot')

    #adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([1, 2],
                    labels=['Pos', 'Neg'])
        ax.set_ylabel('Prob. Distributions')

    #saving image
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


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
model.load_state_dict(torch.load(args.checkpoint, map_location = device, weights_only = False)['model'])
model = model.to(device)

#Calculating and showing AUC scores 
if args.per_class: #it shows AUC per each single class -> we can detect poorly managed classes
    all_probs, all_labels, auc_scores = validate(model, test_dl, device, average = False)
    class_scores = {c:s for (c,s) in zip(CLASSES, auc_scores)} #creating a readable dictionary {class: auc score}

    print(class_scores)

    #plotting output prob distributions for target class
    for target_class in range(len(CLASSES)):
        pos_samples, neg_samples = get_class_dist(all_probs, all_labels, target_class)
        plot_class_dist(pos_samples, neg_samples, out_path = f"dist_{CLASSES[target_class]}.png") 

else:
    _, _, auc_score = validate(model, test_dl, device, average = True) #average AUC over classes
    
    print(auc_score)

