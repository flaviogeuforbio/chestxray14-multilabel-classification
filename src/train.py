from dataset import ChestXRayDataset, create_split, extract_list, CLASSES
from utils import get_pos_weights
from model import ResNet50
from preprocess import train_transform, test_transform
from eval import validate
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torch
import os
import argparse
from tqdm import tqdm

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#parsing CLI options
parser = argparse.ArgumentParser()
parser.add_argument( #dataset root folder path
    "--data_root",
    type = str,
    required = True,
    help = "Path to ChestXRay dataset folder"
)
parser.add_argument( #checkpoints folder path
    "--checkp_dir",
    type = str,
    required = True, 
    help = "Path to Checkpoint models folder"
)
parser.add_argument(
    "--n_epochs",
    type = int,
    required = True,
    help = "Number of training epochs"
)
parser.add_argument(
    "--patience",
    type = int, 
    required = False,
    help = "Early stopping parameter"
)
args = parser.parse_args()

root_dir = args.data_root #getting main folder path by CLI 
csv_name = "Data_Entry_2017_v2020.csv"
train_val_list_name = "train_val_list.txt"
test_list_name = "test_list.txt"
checkpoint_path = os.path.join(args.checkp_dir, "best_model.pt")

#Extracting idxs lists
train_val_list = extract_list(os.path.join(root_dir, train_val_list_name))
test_list = extract_list(os.path.join(root_dir, test_list_name))

#Importing the datasets
train_list, val_list = create_split(train_val_list, val_size = 0.2)

#Building datasets and dataloaders
train_data = ChestXRayDataset(
    img_dir = os.path.join(root_dir, "images"),
    csv_path = os.path.join(root_dir, csv_name),
    list_index = train_list,
    transform = train_transform()
)

val_data = ChestXRayDataset(
    img_dir = os.path.join(root_dir, "images"),
    csv_path = os.path.join(root_dir, csv_name),
    list_index = val_list,
    transform = test_transform() 
)

test_data = ChestXRayDataset(
    img_dir = os.path.join(root_dir, "images"),
    csv_path = os.path.join(root_dir, csv_name),
    list_index = test_list,
    transform = test_transform()
)

train_dl = DataLoader(train_data, batch_size = 64, num_workers = 4, pin_memory = True, persistent_workers = True, shuffle = True)
val_dl = DataLoader(val_data, batch_size = 64, num_workers = 4, pin_memory = True, persistent_workers = True, shuffle = False)
test_dl = DataLoader(test_data, batch_size = 64, num_workers = 4, pin_memory = True, persistent_workers = True, shuffle = False)

#creating model, optimizer, scheduler and defining loss
model = ResNet50(num_classes = 14).to(device)
optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": 1e-3}, #higher lr for head classifier
    {"params": model.backbone.layer4.parameters(), "lr": 1e-4} #lower lr for last convolutional block (fine-tuning)
    ],
    weight_decay = 1e-4
)

scheduler = CosineAnnealingWarmRestarts( #this scheduler allows the net to find a more stable and flat minimum
    optimizer = optimizer,
    T_0 = 5, #5 epochs cycle 
    T_mult = 2,
    eta_min = 1e-6
)

criterion = BCEWithLogitsLoss( #numerically more stable than sigmoid + BCE
    weight = get_pos_weights(train_data).to(device)
)


# training loop
n_epochs = args.n_epochs #parsing from CLI
patience = args.patience if args.patience is not None else 3 #early stopping hyperparameter
best_auc = 0.0 #initializing checkpointing variable
epochs_without_improvement = 0 #counting the plateau time

for i in range(n_epochs):
    train_loss = 0.0

    #progress bar
    pbar = tqdm(
        train_dl,
        desc=f"Epoch [{i+1}/{n_epochs}]",
        leave=False,         
        dynamic_ncols=True
    )

    for images, labels in pbar:
        model.train()
        
        #moving data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        #calculating logits
        logits = model(images)

        #calculating batch loss
        loss = criterion(logits, labels)
        train_loss += loss.item()
        
        #performing the backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": optimizer.param_groups[0]["lr"]
        })

    #calculate general avg loss for training dataset and validation dataset + auc_score on validation
    train_loss /= len(train_dl)
    _, _, val_loss, auc_score = validate(model, val_dl, device, criterion)

    #printing results
    print(f"Epoch {i + 1}: train_loss -> {loss.item()}, validation_loss -> {val_loss}, validation_auc -> {auc_score}")

    #checkpointing
    if auc_score > best_auc:
        epochs_without_improvement = 0

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": i,
            "best_auc": auc_score
            },
            checkpoint_path
        )

        best_auc = auc_score

    else: #in case of plateau or decreasing metrics
        epochs_without_improvement += 1

        if epochs_without_improvement == patience: #early stopping
            print("\n[Early stopping triggered]")
            break

    scheduler.step() #updating lr

