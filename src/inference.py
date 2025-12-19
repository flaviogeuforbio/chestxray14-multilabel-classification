import argparse
from dataset import CLASSES
from model import ResNet50
from preprocess import test_transform
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#parsing data from CLI
parser = argparse.ArgumentParser()
parser.add_argument( #external chest-xray image
    "--image",
    type = str, 
    required = True,
    help = "Path to external image"
)
parser.add_argument( #checkpoint model to use
    "--checkpoint",
    type = str, 
    required = True, 
    help = "Path to checkpoint model to use"
)
parser.add_argument(
    "--gradcam",
    action = "store_true",
    help = "Generate Grad-CAM heatmap"
)
args = parser.parse_args()

def preprocess(image_path, transform, device):
    #open the image
    image = Image.open(image_path).convert("RGB")

    #preprocessing the image
    image = transform(image)

    #moving to device and changing dimensions (single input must have dimension [1, 3, 224, 224])
    image = image.unsqueeze(0).to(device)

    return image 

@torch.no_grad()
def get_probs(image_path: str, model: ResNet50, transform, device):
    #preprocessing image
    image = preprocess(image_path, transform, device)

    #forward pass
    output = model(image)
    probs = torch.sigmoid(output).squeeze(0)

    return probs


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self): #to attach hooks to target layer in the model (layer4)
        def forward_hook(model, input, output):
            self.activations = output.detach()

        def backward_hook(model, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image, target_class): #to generate the heatmap
        #forward pass
        output = self.model(image)

        #isolating target class score and backprop
        score = output[:, target_class]
        score.backward()

        #calculating weighted activations for final layer
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * self.activations, dim = 1))

        #normalizing heatmap
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam 


def stretch_cam(cam, input_tensor): #to stretch heatmap to input image dimension
    cam = cam.unsqueeze(1)
    cam = F.interpolate(
        cam,
        size = input_tensor.shape[2:],
        mode = "bilinear",
        align_corners = True
    )

    return cam

def load_thresholds(thr_path: str): #to load thresholds json file
    with open(thr_path, "r") as f:
        data = json.load(f)
    
    thresholds = data['thresholds']

    return thresholds

def calculate_rel_m(p, t, eps=1e-12): #relative margin (btw raw output prob. and thresholds)
    if p >= t:
        return (p - t) / max(1 - t, eps)
    else:
        return (p - t) / max(t, eps)

def print_probs_preds(probs, thresholds): #to print inference results based on model output + calibrated thresholds
    rel_margins = []

    for i, cls in enumerate(CLASSES):
        p = probs[i] #output prob for class i
        t = thresholds[CLASSES[i]] #calibrated threshold for class i
        abs_m = p - t #absolute margin
        pred = 'POS' if abs_m >= 0 else 'NEG'
        rel_m = calculate_rel_m(p, t) #relative margin

        rel_margins.append(rel_m)

        print(f"{cls:18s} "
          f"prob={p:5.3f} | "
          f"thr={t:5.3f} | "
          f"{pred:3s} | "
          f"abs={abs_m:+6.3f} | "
          f"rel={rel_m:+6.3f}")
        
    return rel_margins

def get_target_classes(rel_margins: list, default_n = 1): #to get target classes for GradCAM analysis
    rel_margins = np.array(rel_margins)

    #case all predicted negatives
    if np.all(rel_margins < 0):
        idx = list(np.argsort(rel_margins))

        return idx[-default_n:]
    
    #case some POS prediction
    else:
        idxs = []
        for i, m in enumerate(rel_margins):
            if m >= 0:
                idxs.append(i)

        return idxs


#defining model and preprocessing transform
model = ResNet50()
model.load_state_dict(torch.load(args.checkpoint, map_location = device)['model'])
model = model.to(device) #GPU compatibility
model.eval() #BN in eval mode
transform = test_transform()

#getting output probabilities
probs = get_probs(args.image, model, transform, device)

#using thresholds (saved in json file after calibration) to predict classes and print results
thresholds = load_thresholds("thresholds.json") #hard-coded for now
rel_margins = print_probs_preds(probs, thresholds) #collects relative margins too for GradCAM analysis if needed


#GradCAM analysis (working progress: add compatibility with thresholds)
if args.gradcam:
    #preprocessing image and finding target classes (based on decision margins)
    image = preprocess(args.image, transform, device)
    # target_class = torch.argmax(probs, dim = -1).item()
    target_classes = get_target_classes(rel_margins)

    Path("./gradcam_analysis").mkdir(parents=True, exist_ok=True) #create GradCAM heatmaps folder if non-existent

    #creating heatmaps
    gradcam = GradCAM(model, model.backbone.layer4)
    image_toplot = image.cpu().detach().numpy()

    for cls in target_classes:
        cam = gradcam.generate(image, cls)
        cam = stretch_cam(cam, image)

        #visualizing & saving input image + heatmap
        cam = cam.cpu().detach().numpy()

        #create a new figure for each class
        plt.figure(figsize=(6, 6))
        plt.title(f"{CLASSES[cls]}")
        plt.imshow(image_toplot[0].transpose(1, 2, 0))
        plt.imshow(cam[0, 0], cmap = 'jet', alpha = 0.5)
        plt.axis('off')

        out_path = f"gradcam_analysis/gradcam_{CLASSES[cls]}.png"
        plt.savefig(fname = out_path, dpi = 200, bbox_inches = "tight")
        plt.close()
        print(f"Saved at {out_path}")