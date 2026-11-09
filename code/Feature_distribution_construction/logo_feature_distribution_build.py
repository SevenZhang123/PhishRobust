# Construct the logo feature distribution for each brand.

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from common_utils import LogoEncoder, get_transform
from PIL import Image

class DistributionBuilder:
    def __init__(self, data_root, model_path, save_path, brand_list, device='cuda'):
        self.data_root = data_root
        self.save_path = save_path
        self.device = device
        self.brand_list = brand_list

        self.model = LogoEncoder().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.transform = get_transform()

        # Covariance regularization coefficient (to prevent matrix non-invertibility)
        self.regularization = 0.001

    def extract_features(self, img_path):
        """Extracting features from a single image"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(img_tensor)
            return feat.cpu().numpy().squeeze()  # (2048,)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None

    def build(self):
        brand_stats = {}  # Store the final result

        # Iterate through the folders corresponding to 134 brands
        for brand in tqdm(self.brand_list):
            brand_dir = os.path.join(self.data_root, brand)
            features = []

            # Iterate through all images under this brand
            for img_name in os.listdir(brand_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if img_name.lower() == "homepage.png" or img_name.lower() == "loginpage.png":
                    continue

                img_path = os.path.join(brand_dir, img_name)
                feat = self.extract_features(img_path)
                if feat is not None:
                    features.append(feat)

            features = np.array(features)  # Shape: (N, 2048)

            if len(features) < 2:
                print(f"[Warning] Brand {brand} has less than 2 images. Skipping covariance calculation.")
                continue

            # --- Statistical modeling ---

            # 1. Calculate the mean vector mu (2048,)
            mu = np.mean(features, axis=0)

            # 2. Calculate the covariance matrix Sigma (2048, 2048)
            # rowvar=False indicates that each column is a characteristic variable.
            sigma = np.cov(features, rowvar=False)

            # 3. Regularization (Sigma + lambda * I)
            # This step ensures that the matrix is ​​invertible (non-singular).
            sigma_reg = sigma + self.regularization * np.eye(sigma.shape[0])

            # 4. Pre-compute the inverse matrix (Precision Matrix)
            # Using the inverse matrix directly during inference is faster
            try:
                sigma_inv = np.linalg.inv(sigma_reg)
            except np.linalg.LinAlgError:
                # If it is still irreversible, use pseudo-reverse
                sigma_inv = np.linalg.pinv(sigma_reg)

            brand_stats[brand] = {
                'mu': mu,
                'sigma_inv': sigma_inv,
                'n_samples': len(features)
            }

        # Save the distribution library
        with open(self.save_path, 'wb') as f:
            pickle.dump(brand_stats, f)

        print(f"Distribution database saved to {self.save_path}")


if __name__ == "__main__":
    # Config
    DATA_ROOT = '../../data/target_list_134/target_list_134_augmented'
    MODEL_PATH = '../../models/robust_logo_encoder.pth'
    SAVE_DB_PATH = '../../models/brand_logo_feature_distribution.pkl'

    brand_list = []
    f_brand = open("../../data/target_list_134/target_dict.txt", "r")
    for line in f_brand:
        brand_list.append(line.split("\t")[0])

    builder = DistributionBuilder(DATA_ROOT, MODEL_PATH, SAVE_DB_PATH, brand_list)
    builder.build()