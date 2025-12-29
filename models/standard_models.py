from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import resnet50, ResNet50_Weights

#------------------------------------
# standard transfer learning models |
#------------------------------------

# ViT-Base\16
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Sequential(
    nn.Dropout(0.1),  
    nn.Linear(model.heads.head.in_features, 2)
)

# ConvNeXt-Tiny
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.classifier[2].in_features, 2)
)

# ResNet-50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 2)
)
