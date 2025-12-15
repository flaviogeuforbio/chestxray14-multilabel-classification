import argparse
from dataset import CLASSES
from model import ResNet50
from preprocess import test_transform
from PIL import Image
import torch

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
args = parser.parse_args()

@torch.no_grad()
def get_probs(image_path: str, model: ResNet50, transform, device):
    #open the image
    image = Image.open(image_path).convert("RGB")

    #preprocessing the image
    image = transform(image)

    #moving to device and changing dimensions (single input must have dimension [1, 3, 224, 224])
    image = image.unsqueeze(0).to(device)

    #forward pass
    model.eval() #BN in eval mode
    output = model(image)
    probs = torch.sigmoid(output).squeeze(0)

    return probs


#defining model and preprocessing transform
model = ResNet50()
model.load_state_dict(torch.load(args.checkpoint)['model']).to(device)
transform = test_transform()

#getting output probabilities
probs = get_probs(args.image, model, transform, device)
class_probs = {c:float(p.item()) for (c,p) in zip(CLASSES, probs)} #creating a readable dictionary {class: probability}

#printing out results
print(class_probs)