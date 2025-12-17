import argparse
from dataset import CLASSES
from model import ResNet50
from preprocess import test_transform
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def preprocess(image_path):
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
    image = preprocess(image_path)

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


#defining model and preprocessing transform
model = ResNet50()
model.load_state_dict(torch.load(args.checkpoint, map_location = device)['model'])
model = model.to(device) #GPU compatibility
model.eval() #BN in eval mode
transform = test_transform()

#getting output probabilities
probs = get_probs(args.image, model, transform, device)
class_probs = {c:float(p.item()) for (c,p) in zip(CLASSES, probs)} #creating a readable dictionary {class: probability}

#printing out results
print(class_probs)


#GradCAM analysis
if args.gradcam:
    #preprocessing image and finding target class (as most probable)
    image = preprocess(args.image)
    target_class = torch.argmax(probs, dim = -1).item()

    #creating heatmap
    gradcam = GradCAM(model, model.backbone.layer4)
    cam = gradcam.generate(image, target_class)
    cam = stretch_cam(cam, image)

    #visualizing input image + heatmap
    cam = cam.cpu().detach().numpy()
    image = image.cpu().detach().numpy()

    plt.imshow(image[0].transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

    plt.imshow(cam[0].transpose(1, 2, 0), cmap = 'jet', alpha = 0.5)
    plt.axis('off')
    plt.show()