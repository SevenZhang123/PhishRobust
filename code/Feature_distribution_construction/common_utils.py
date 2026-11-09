# Define model structure and data preprocessing methods.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps

# ==========================================
# 1. Model Definition (LogoEncoder)
# ==========================================
class LogoEncoder(nn.Module):
    def __init__(self):
        super(LogoEncoder, self).__init__()
        base_model = models.convnext_tiny(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.projector = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        embedding = self.projector(x)
        # Perform L2 normalization
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        return norm_embedding

# ==========================================
# 2. Preprocessing definition
# ==========================================
class SquarePad:
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padding = ((max_side - w) // 2, (max_side - h) // 2, (max_side - w + 1) // 2, (max_side - h + 1) // 2)
        return ImageOps.expand(img, padding, fill=(255, 255, 255))

def get_transform():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])