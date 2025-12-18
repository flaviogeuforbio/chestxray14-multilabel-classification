from sklearn.metrics import roc_auc_score 
import torch
from tqdm import tqdm 

@torch.no_grad()
def validate(model, val_dl, device, criterion = None, average = True):
    model.eval()

    loss = 0
    all_probs = []
    all_labels = []

    #progress bar
    pbar = tqdm(
        val_dl,
        desc=f"Validation: ",
        leave=False,         
        dynamic_ncols=True
    )

    for i, (images, labels) in enumerate(val_dl):
        #moving data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        #calculating logits
        logits = model(images)
        
        #calculating batch loss and updating running_loss
        if criterion is not None:
            running_loss = criterion(logits, labels).item()
            loss += running_loss

        #calculating batch probability dist
        probs = torch.sigmoid(logits)

        #updating arrays
        all_probs.append(probs)
        all_labels.append(labels)

        #update progress bar
        if criterion is not None:
            pbar.set_postfix({
                "loss": f"{running_loss.item():.4f}",
            })

    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    if average:
        auc_score = roc_auc_score(
            all_labels,
            all_probs,
            average = "macro"
        )
    else:
        auc_score = roc_auc_score(
            all_labels,
            all_probs,
            average = None
        )

    if criterion is not None:
        return all_probs, all_labels, loss / len(val_dl), auc_score

    else:
        return all_probs, all_labels, auc_score
        
