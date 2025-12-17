from sklearn.metrics import roc_auc_score 
import torch

@torch.no_grad()
def validate(model, val_dl, device, criterion = None, average = True):
    model.eval()

    loss = 0
    all_probs = []
    all_labels = []

    for images, labels in val_dl:
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
        probs = torch.sigmoid(logits, dim=1)

        #updating arrays
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = torch.cat(all_probs).cpu()
    all_labels = torch.cat(all_labels).cpu()

    if average:
        auc_score = roc_auc_score(
            all_labels.numpy(),
            all_probs.numpy(),
            average = "macro"
        )
    else:
        auc_score = roc_auc_score(
            all_labels.numpy(),
            all_probs.numpy(),
            average = None
        )

    if criterion is not None:
        return loss / len(val_dl), auc_score

    else:
        return auc_score
        
