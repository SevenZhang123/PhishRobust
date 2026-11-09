# PhishRobust's overall process
# PhishRobust input: website screenshot
# PhishRobust process:
# 1. Logo location and extraction.
# 2. Logo brand recognition.
# 3. Brand-domain matching.
#
# 1. Logo localization and extraction: Logo localization and extraction based on YOLO11-Detect

# 2. Logo Brand Recognition: First, a robust logo encoder is used to encode the logo extracted in the first step.
# Then, the brand corresponding to the logo is determined based on the logo feature distribution of each brand.

# 3. After obtaining the target website's brand in step 2,
# we use our constructed reference-list to determine whether the website's domain
# name is in the brand's corresponding domain name list, thus completing the phishing website detection.

import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageOps
from ultralytics import YOLO
from urllib.parse import urlparse
from datetime import datetime


# ==========================================
# 1. Configuration Parameters
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR_ROOT = '../data'
    MODEL_DIR_ROOT = '../models'

    # The path of models
    YOLO_MODEL_PATH = os.path.join(MODEL_DIR_ROOT, 'logo_location_and_select.pt')
    ENCODER_MODEL_PATH = os.path.join(MODEL_DIR_ROOT, 'robust_logo_encoder.pth')
    DISTRIBUTION_DB_PATH = os.path.join(MODEL_DIR_ROOT, 'brand_logo_feature_distribution.pkl')

    # The path of data
    REF_LIST_ROOT = os.path.join(DATA_DIR_ROOT, 'brand_domain')

    YOLO_CONF_THRES = 0.4
    FEATURE_DIM = 2048  # ConvNeXt Tiny Projector output
    CONFIDENCE_LEVEL = 0.95  # Chi-square distribution reliability


# ==========================================
# 2. Auxiliary Tools and Model Definition
# ==========================================

# 2.1 Custom Transform (SquarePad)
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


# 2.2 Robust Logo Encoder Definition
# Corresponding to the "Model Transfer" section of the paper:
# Remove the classification head, retain the projector, and output the L2 normalized vector.
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
        # Forced L2 normalization
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        return norm_embedding


# ==========================================
# 3. PhishRobust System Core Classes
# ==========================================
class PhishRobustSystem:
    def __init__(self):
        print("[System] Initializing PhishRobust...")
        self.device = Config.device

        # 1. Load the YOLO model (Logo positioning)
        print(f"[Module 1] Loading YOLO model from {Config.YOLO_MODEL_PATH}...")
        self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)

        # 2. Load Logo Encoder (Brand Recognition)
        print(f"[Module 2] Loading Robust Logo Encoder from {Config.ENCODER_MODEL_PATH}...")
        self.encoder = LogoEncoder().to(self.device)

        state_dict = torch.load(Config.ENCODER_MODEL_PATH, map_location=self.device)
        self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder.eval()
        self.transform = get_transform()

        # 3. Load the brand feature distribution library (similarity measurement)
        print(f"[Module 2] Loading Feature Distribution DB from {Config.DISTRIBUTION_DB_PATH}...")
        with open(Config.DISTRIBUTION_DB_PATH, 'rb') as f:
            self.brand_db = pickle.load(f)

        # Calculate the chi-square threshold
        # self.threshold = chi2.ppf(Config.CONFIDENCE_LEVEL, df=Config.FEATURE_DIM)
        self.threshold = 40
        print(f" Mahalanobis Threshold (alpha={Config.CONFIDENCE_LEVEL}): {self.threshold:.2f}")

        # 4. Load the Knowledge Base (Brand-Domain Matching)
        print(f"[Module 3] Loading Domain List from {Config.REF_LIST_ROOT}...")
        self.reference_lists = self._load_reference_lists()
        print(f" Loaded domains for {len(self.reference_lists)} brands.")

    def _load_reference_lists(self):
        ref_dict = {}
        if not os.path.exists(Config.REF_LIST_ROOT):
            print(f"Warning: Reference list root {Config.REF_LIST_ROOT} does not exist.")
            return ref_dict

        for file in Path(Config.REF_LIST_ROOT).iterdir():
            if file.is_file():
                brand_name = file.stem.split('_domain_list')[0]
                file_path = os.path.join(Config.REF_LIST_ROOT, file)

                domains = set()
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            dom = line.strip().lower()
                            if dom:
                                domains.add(dom)

                ref_dict[brand_name] = domains

        return ref_dict

    def extract_effective_domain(self, url):
        """
        Extract the main domain from the URL (e.g., 'https://www.google.com/xyz' -> 'google.com')
        """
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain
        except:
            return url

    # ==========================================
    # Core Flow Functions
    # ==========================================

    def step1_detect_logo(self, img_path):
        """
        Step 1: Logo Location and Extraction
        Returns: A PIL Image object (cropped Logo) or None
        """
        try:
            results = self.yolo_model.predict(source=img_path, conf=Config.YOLO_CONF_THRES, verbose=False)

            best_crop = None
            max_conf = -1

            result = results[0]
            if len(result.boxes) == 0:
                return None

            # Load original image
            orig_img = Image.open(img_path).convert('RGB')
            w, h = orig_img.size

            for box in result.boxes:
                conf = float(box.conf[0].cpu())
                if conf > max_conf:
                    max_conf = conf
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    # Boundary checks
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        best_crop = orig_img.crop((x1, y1, x2, y2))

            return best_crop
        except Exception as e:
            print(f"Error in logo detection: {e}")
            return None

    def _calculate_mahalanobis(self, z, mu, sigma_inv):
        delta = z - mu
        left = np.dot(delta, sigma_inv)
        dist_sq = np.dot(left, delta)
        return dist_sq

    def step2_identify_brand(self, logo_img):
        """
        Step 2: Logo Brand Recognition
        Return: (BrandName, Distance) or (None, inf)
        """
        if logo_img is None:
            return None, float('inf')

        img_tensor = self.transform(logo_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.encoder(img_tensor).cpu().numpy().squeeze()

        # 2. Similarity measurement
        min_dist = float('inf')
        best_brand = None

        for brand, stats in self.brand_db.items():
            # Calculate Mahalanobis distance
            dist = self._calculate_mahalanobis(embedding, stats['mu'], stats['sigma_inv'])
            if dist < min_dist:
                min_dist = dist
                best_brand = brand

        # 3. Threshold determination
        if min_dist < self.threshold:
            return best_brand, min_dist
        else:
            return None, min_dist

    def step3_match_domain(self, brand, url):
        """
        Step 3: Brand-Domain Consistency Verification
        Returns: True (Benign), False (phishing)
        """
        if brand is None:
            # If no brand is identified, logically:
            # 1. Since there is no visual deception (no logo or unrecognizable logo),
            # it is usually judged as Benign (0).
            # 2. Or it can be classified as Unknown.
            # Here we follow the common logic: if no visual features of the protected brand are detected,
            # it is not considered a phishing attempt against that brand.
            return True

        brand = brand.replace(" ", "_")

        if brand not in self.reference_lists:
            # The brand was identified, but it is not in the knowledge base (anomaly).
            print(f"Warning: Identified brand '{brand}' not in domain list keys.")
            return True  # Conservative strategy: Consider it a benign website

        # Extract target domain
        target_domain = self.extract_effective_domain(url)

        # Check if it's on the whitelist
        whitelist = self.reference_lists[brand]

        # If the whitelist contains 'google.com', then 'accounts.google.com' should be benign.
        for white_dom in whitelist:
            if target_domain == white_dom or target_domain.endswith('.' + white_dom):
                return True  # benign

        return False  # The domain is not on the whitelist and the page contains the brand's logo -> Phishing!

    def predict(self, img_path, url):
        """
        Single prediction entry
        Returns:
            label (int): 1 (Phishing), 0 (Benign)
            info (dict): Intermediate results
        """
        # Step 1
        logo_crop = self.step1_detect_logo(img_path)

        # Step 2
        detected_brand, dist = self.step2_identify_brand(logo_crop)

        # Step 3
        is_legitimate = self.step3_match_domain(detected_brand, url)

        # Final judgment：
        # If Step 2 identifies the brand and Step 3 determines the domain name is invalid -> phishing (1)
        # Otherwise -> Benign (0)

        if detected_brand is not None and not is_legitimate:
            prediction = 1
        else:
            prediction = 0

        return prediction, {
            "logo_found": logo_crop is not None,
            "brand": detected_brand,
            "dist": dist,
            "url": url
        }


# ==========================================
# 4. Batch Evaluation Process
# ==========================================
def run_evaluation(label_file_path):
    system = PhishRobustSystem()

    label_file = label_file_path
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        return

    print(f"Reading {label_file}...")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = open(f"./{current_time}_results.txt", "w", encoding="utf-8")

    total = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    t0 = time.time()

    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue

        parts = line.split("\t")
        rel_path = parts[0]
        simulated_url = parts[1]
        gt_label = int(parts[2])

        full_img_path = os.path.join(Config.DATA_DIR_ROOT, rel_path)

        if not os.path.exists(full_img_path):
            continue

        # --- predict ---
        pred_label, info = system.predict(full_img_path, simulated_url)

        result_file.write(full_img_path + "\t" + info["url"] + "\t" + str(gt_label) + "\t" + str(pred_label) + "\t" + str(info["brand"]) + "\t" + str(info["dist"]) + "\n")

        # --- statistics ---
        if pred_label == gt_label:
            correct += 1
            if gt_label == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred_label == 1:
                fp += 1
            else:
                fn += 1

        total += 1

        if total % 100 == 0:
            print(f"\rProcessed {total} samples...", end="")

    t1 = time.time()

    # --- Output ---
    print(f"\n\nEvaluation Completed in {t1 - t0:.2f}s")
    print("=" * 30)
    print(f"Total Samples: {total}")
    print(f"Accuracy:      {correct / total:.4f}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    average_time = (t1 - t0) / total

    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Average Time:   {average_time:.2f}s")
    print("=" * 30)
    print("Confusion Matrix:")
    print(f"TP: {tp} | FN: {fn}")
    print(f"FP: {fp} | TN: {tn}")


if __name__ == "__main__":

    file_path = 'sample_label.txt'
    label_file_path = os.path.join(Config.DATA_DIR_ROOT, file_path)
    run_evaluation(label_file_path)
