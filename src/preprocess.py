from torchvision import transforms

def train_transform(): #preprocessing pipeline for training data
    return transforms.Compose([
        transforms.Resize((224, 224)), #ResNet wanted input dimension
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # From the ResNet Paper: statistical distribution of training inputs
            std=[0.229, 0.224, 0.225]
        )
    ])

def test_transform(): #preprocessing pipeline for test data
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
    ])

