import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import resnet50, ResNet50_Weights

#---------------
# glass models |
#---------------
class GlobalLocalAttentionModel(nn.Module):
    def __init__(self, dropout_rate=0.1, model_type=None):
        super().__init__()
        
        if model_type == "vit":
            self.global_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.global_model.heads = nn.Identity()

            self.local_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.local_model.heads = nn.Identity()

            self.embedding_dim = 768
            
        elif model_type == "resnet":
            self.global_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.global_model.fc = nn.Identity()

            self.local_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.local_model.fc = nn.Identity()

            self.embedding_dim = 2048
            
        elif model_type == "convnext":
            self.global_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.global_model.classifier = nn.Identity()

            self.local_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.local_model.classifier = nn.Identity()

            self.embedding_dim = 768
            
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # attention mechanism over local crops
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        # classifier for global + local features
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embedding_dim * 2, 2)
        )

    def forward(self, global_view, local_crops):
        # global_view: (B, 3, 224, 224)
        # local_crops: (B, num_crops, 3, 224, 224)
        B = global_view.shape[0]
        num_crops = local_crops.shape[1]
        
        #-----------------
        # global forward |
        #-----------------
        global_feat = self.global_model(global_view) # (B, embedding_dim) for vit and resnet
                                                     # (B, embedding_dim, 1, 1) for convnext

        # for convnext compatibility
        global_feat = global_feat.flatten(start_dim=1) # (B, embedding_dim, 1, 1) -> (B, embedding_dim) for convnext
                                                       # (B, embedding_dim) -> (B, embedding_dim) for vit and resnet

        #----------------
        # local forward |
        #----------------
        local_crops_flat = local_crops.view(-1, *local_crops.shape[2:]) # (B, num_crops, 3, 224, 224) -> (B*num_crops, 3, 224, 224)
        local_feat_flat = self.local_model(local_crops_flat)            # (B*num_crops, embedding_dim) for vit and resnet
                                                                        # (B*num_crops, embedding_dim, 1, 1) for convnext

        # for convnext compatibility
        local_feat_flat = local_feat_flat.flatten(start_dim=1) # (B*num_crops, embedding_dim, 1, 1) -> (B*num_crops, embedding_dim)
        
        local_feat = local_feat_flat.view(B, num_crops, -1) # (B, num_crops, embedding_dim)

        #------------------------------
        # attention on local features |
        #------------------------------
        attention_score = self.attention(local_feat) # (B, num_crops, 1)
        attention_weight = F.softmax(attention_score, dim=1) # (B, num_crops, 1)
        local_aggregated = torch.sum(attention_weight * local_feat, dim=1) # (B, embedding_dim)

        #-----------------------------------------------
        # concat global + local and input to classifier|
        #-----------------------------------------------
        combined = torch.cat([global_feat, local_aggregated], dim=1) # (B, embedding_dim*2)
        logits = self.classifier(combined)
        
        return logits
