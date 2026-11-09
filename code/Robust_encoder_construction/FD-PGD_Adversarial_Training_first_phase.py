# The initial classifier was subjected to a first-stage adversarial training
# based on the logo-2K+ dataset to build a robust classifier: robust_classifier_first_phase.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import logging
import numpy as np
from PIL import Image, ImageOps  # 即使使用 ImageFolder 也需要这个
import json
import pickle
import random

try:
    import torchattacks
except ImportError:
    print("Please install torchattacks first: pip install torchattacks")
    exit()


# ==========================================
# 1. Configuration Parameters
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2338
    batch_size = 64
    learning_rate = 1e-5
    num_epochs = 15

    # Attack parameters
    epsilon = 8 / 255.0
    alpha = 2.5 / 255.0
    steps = 12

    # Mixed attack probability
    fd_pgd_prob = 1

    # Loss weights
    lambda_clean = 0.3
    lambda_adv = 0.7

    data_dir = '../../data/Logo-2K+/train_and_test'
    pretrained_path = "../../models/initial_classifier.pth"
    save_dir = "../../models"


# ==========================================
# 0. Auxiliary Tools
# ==========================================
def setup_logger(log_dir='logfile', log_name='adversarial_train_logo2k'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f'{log_name}.log'))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ==========================================
# Custom Transformation: Maintain Proportional Fill
# ==========================================
class SquarePad:
    """
    Fill the image with squares
    To prevent logo distortion caused by direct resizing
    """

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padding = ((max_side - w) // 2, (max_side - h) // 2, (max_side - w + 1) // 2, (max_side - h + 1) // 2)
        return ImageOps.expand(img, padding, fill=(255, 255, 255))


# ==========================================
# 2. Dataset Loading
# ==========================================
def get_dataloaders():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Introduce SquarePad, first fill it into a square, then resize it.
    data_transforms = {
        "train": transforms.Compose([
            SquarePad(),  # 1. filling
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        "test": transforms.Compose([
            SquarePad(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    if not os.path.exists(os.path.join(Config.data_dir, 'train')):
        print(f"Error: Incorrect dataset path, cannot find {os.path.join(Config.data_dir, 'train')}")
        exit()

    image_datasets = {x: datasets.ImageFolder(os.path.join(Config.data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=Config.batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'test']}

    print(f"Dataset loaded. Classes found: {len(image_datasets['train'].classes)}")

    return dataloaders


# ==========================================
# 3. Model Definition
# ==========================================
class RobustConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(RobustConvNeXt, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        embedding = self.projector(x)
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return logits, norm_embedding


# ==========================================
# 4. Adversarial Sample Generation Function
# ==========================================
def fd_pgd_attack(model, images, epsilon, alpha, steps):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)

    images_01 = images * std + mean
    images_01 = torch.clamp(images_01, 0, 1)

    with torch.no_grad():
        _, clean_norm_embeddings = model(images)
        clean_norm_embeddings = clean_norm_embeddings.detach()

    adv_images_01 = images_01.clone().detach()
    adv_images_01 = adv_images_01 + torch.empty_like(adv_images_01).uniform_(-epsilon, epsilon)
    adv_images_01 = torch.clamp(adv_images_01, 0, 1)

    for _ in range(steps):
        adv_images_01.requires_grad = True
        adv_images_norm = (adv_images_01 - mean) / std
        _, adv_norm_embeddings = model(adv_images_norm)
        cos_sim = F.cosine_similarity(adv_norm_embeddings, clean_norm_embeddings, dim = 1).mean()
        grad = torch.autograd.grad(cos_sim, adv_images_01)[0]
        adv_images_01 = adv_images_01.detach() - alpha * grad.sign()
        delta = torch.clamp(adv_images_01 - images_01, min = -epsilon, max = epsilon)
        adv_images_01 = torch.clamp(images_01 + delta, min = 0, max = 1).detach()

    adv_images_final = (adv_images_01 - mean) / std
    return adv_images_final


def classification_pgd_attack(model, images, labels, epsilon, alpha, steps):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    criterion = nn.CrossEntropyLoss()

    images_01 = images * std + mean
    images_01 = torch.clamp(images_01, 0, 1)

    adv_images_01 = images_01.clone().detach()
    adv_images_01 = adv_images_01 + torch.empty_like(adv_images_01).uniform_(-epsilon, epsilon)
    adv_images_01 = torch.clamp(adv_images_01, 0, 1)

    for _ in range(steps):
        adv_images_01.requires_grad = True
        adv_images_norm = (adv_images_01 - mean) / std
        logits, _ = model(adv_images_norm)
        loss = criterion(logits, labels)
        grad = torch.autograd.grad(loss, adv_images_01)[0]
        adv_images_01 = adv_images_01.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images_01 - images_01, min = -epsilon, max = epsilon)
        adv_images_01 = torch.clamp(images_01 + delta, min = 0, max = 1).detach()

    adv_images_final = (adv_images_01 - mean) / std
    return adv_images_final


# ==========================================
# 5. Robustness Evaluation Function
# ==========================================
def evaluate_robustness(model, dataloader, device):
    model.eval()

    class TorchAttackModelWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, x):
            logits, _ = self.model(x)
            return logits

    attack_model = TorchAttackModelWrapper(model)
    attack_model.eval()

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    attacks = {
        'Clean': None,
        'FGSM': torchattacks.FGSM(attack_model, eps=8 / 255),
        'PGD-10': torchattacks.PGD(attack_model, eps=8 / 255, alpha=2 / 255, steps=10),
    }

    for name, atk in attacks.items():
        if atk is not None:
            atk.set_normalization_used(mean=norm_mean, std=norm_std)

    results = {}
    limit_batches = 10

    for name, atk in attacks.items():
        correct = 0
        total = 0
        total_cos_sim = 0

        for i, (images, labels) in enumerate(dataloader):
            if i >= limit_batches: break

            images, labels = images.to(device), labels.to(device)

            if atk is None:
                adv_images = images
            else:
                mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(device)
                std = torch.tensor(norm_std).view(1, 3, 1, 1).to(device)
                images_01 = images * std + mean
                adv_images_01 = atk(images_01, labels)
                adv_images = (adv_images_01 - mean) / std

            with torch.no_grad():
                _, norm_embedding_clean = model(images)
                logits_adv, norm_embedding_adv = model(adv_images)
                _, preds = logits_adv.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                if name != 'Clean':
                    cos_sim = F.cosine_similarity(norm_embedding_clean, norm_embedding_adv, dim=1).mean()
                    total_cos_sim += cos_sim.item() * labels.size(0)

        acc = 100 * correct / total
        results[f'{name}_Acc'] = acc
        if name != 'Clean':
            avg_cos_sim = total_cos_sim / total
            results[f'{name}_Sim'] = avg_cos_sim
        else:
            results[f'{name}_Sim'] = 1.0
    return results


# ==========================================
# 6. Main Training Process
# ==========================================
def train_robust_model():
    logger = setup_logger(log_name='adversarial_train_logo2k')
    loaders = get_dataloaders()
    model = RobustConvNeXt(Config.num_classes).to(Config.device)

    if os.path.exists(Config.pretrained_path):
        logger.info(f"Loading pre-trained weights: {Config.pretrained_path}")
        state_dict = torch.load(Config.pretrained_path, map_location=Config.device)

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        removed_keys = [k for k in state_dict.keys() if k not in pretrained_dict]
        logger.info(f"Ignore mismatched layers (reinitialize): {removed_keys}")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        logger.error("Pre-trained weights do not exist")
        exit(1)

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    logger.info(f"Start Logo-2K+ Hybrid Attack Training | Classes: {Config.num_classes} | Steps: {Config.steps}")

    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0
        total_clean_acc = 0
        total_adv_acc = 0
        samples = 0
        start_time = time.time()

        fd_count = 0
        cls_count = 0

        for i, (clean_images, labels) in enumerate(loaders['train']):
            clean_images, labels = clean_images.to(Config.device), labels.to(Config.device)
            batch_size = labels.size(0)

            # --- Step 1: Adversarial sample generation ---
            if random.random() < Config.fd_pgd_prob:
                adv_images = fd_pgd_attack(model, clean_images,
                                           Config.epsilon, Config.alpha, Config.steps)
                fd_count += 1
            else:
                adv_images = classification_pgd_attack(model, clean_images, labels,
                                                       Config.epsilon, Config.alpha, Config.steps)
                
                cls_count += 1

            # --- Step 2: Intensive training ---
            model.train()
            optimizer.zero_grad()

            logits_clean, _ = model(clean_images)
            loss_clean = criterion(logits_clean, labels)

            logits_adv, _ = model(adv_images)
            loss_adv = criterion(logits_adv, labels)

            loss = Config.lambda_clean * loss_clean + Config.lambda_adv * loss_adv

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_clean_acc += (logits_clean.argmax(1) == labels).sum().item()
            total_adv_acc += (logits_adv.argmax(1) == labels).sum().item()
            samples += batch_size

            if (i + 1) % 10 == 0:
                print(f"\rEpoch {epoch + 1} Batch {i + 1}/{len(loaders['train'])} Loss: {loss.item():.4f}", end="")

        avg_loss = total_loss / samples
        avg_clean_acc = 100 * total_clean_acc / samples
        avg_adv_acc = 100 * total_adv_acc / samples

        logger.info(f"\n[Epoch {epoch + 1}] Time: {(time.time() - start_time) / 60:.1f}m | Loss: {avg_loss:.4f}")
        logger.info(f"Train Clean Acc: {avg_clean_acc:.2f}% | Train Adv Acc (Hybrid): {avg_adv_acc:.2f}%")
        logger.info(f"Attack Mix: FD-PGD={fd_count} batches | CLS-PGD={cls_count} batches")

        # Evaluate
        logger.info(">>> Running Robustness Evaluation...")
        eval_results = evaluate_robustness(model, loaders['test'], Config.device)
        res_str = (
            f"Clean Acc: {eval_results['Clean_Acc']:.2f}% | "
            f"FGSM Acc: {eval_results['FGSM_Acc']:.2f}% | "
            f"PGD-10 Acc: {eval_results['PGD-10_Acc']:.2f}% || "
            f"FGSM Sim: {eval_results['FGSM_Sim']:.4f} | "
            f"PGD-10 Sim: {eval_results['PGD-10_Sim']:.4f}"
        )
        logger.info(f"Test Evaluation: {res_str}")

        if eval_results['PGD-10_Acc'] > best_acc:
            best_acc = eval_results['PGD-10_Acc']
            torch.save(model.state_dict(), os.path.join(Config.save_dir, "robust_classifier_first_phase.pth"))
            logger.info(f"[*] New Best Model Saved!")

if __name__ == "__main__":
    train_robust_model()