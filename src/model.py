from torchvision.models import resnet50
from torch import nn


class ResNet50(nn.Module):
    def __init__(self, num_classes = 14):
        super().__init__()

        #importing the pre-trained ResNet50 (ImageNet dataset recognition)
        self.backbone = resnet50(weights='DEFAULT')

        #Substituting just the final fc layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        #creating a new classifier (14 classes)
        self.classifier = nn.Linear(in_features, num_classes)

        #freezing the layers except for the last one (to fine-tune the model)
        for name, param in self.backbone.named_parameters():
            if not name.startswith('layer4'):
                param.requires_grad = False

    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits