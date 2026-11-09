# Based on the robust classifier `robust_classifier_logo2K2targelist.pth` trained adversarially previously,
# a robust logo encoder is constructed through transfer learning.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


class LogoEncoder(nn.Module):
    """
    The final robust logo encoder
    Input:  Image Tensor (B, 3, 224, 224)
    Output: Normalized Embedding (B, 2048)
    """

    def __init__(self):
        super(LogoEncoder, self).__init__()
        # 1. Backbone
        base_model = models.convnext_tiny(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # 2. Projection Head
        # 768 -> 2048
        self.projector = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)

        # Projected into higher-dimensional space
        embedding = self.projector(x)

        # L2 normalization
        norm_embedding = F.normalize(embedding, p=2, dim=1)

        return norm_embedding


def migrate_model():
    classifier_path = "../../models/robust_classifier_second_phase.pth"
    encoder_save_path = "../../models/robust_logo_encoder.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Step 1] Load the trained classifier weights: {classifier_path}")
    if not os.path.exists(classifier_path):
        print("Error: Classifier weight file not found!")
        return

    # Load the complete classifier weight dictionary
    classifier_state_dict = torch.load(classifier_path, map_location=device)

    # Instantiate an empty encoder
    encoder = LogoEncoder().to(device)
    encoder_state_dict = encoder.state_dict()

    # [Step 2] Weighted Filtering and Migration
    # Logic: Only retain the weights of the backbone (features, avgpool) and projector.
    # Discard the weight of the classifier
    new_state_dict = {}
    for k, v in classifier_state_dict.items():
        if k in encoder_state_dict:
            if v.shape == encoder_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Warning: Layer {k} dimension mismatch, skipped.")
        else:
            pass

    # [Step 3] Load the filtered weights
    encoder.load_state_dict(new_state_dict)
    print(f"The {len(new_state_dict)} layer parameter was successfully migrated.")
    print(f"The classifier header parameter was discarded.")

    # [Step 4] Save the final encoder
    torch.save(encoder.state_dict(), encoder_save_path)
    print(f"The logo encoder has been saved to:{encoder_save_path}")

    encoder.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = encoder(dummy_input)

    print("\n[Verification output]")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")  # It should be [1, 2048]
    print(f"Output Norm: {torch.norm(output, p=2).item():.4f}")  # Should be close 1.0


if __name__ == "__main__":
    migrate_model()
