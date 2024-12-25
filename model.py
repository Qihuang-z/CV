import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B7_Weights

def get_model(num_classes):
    model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
