# The first step in building the logo encoder is initial training:
# the data used consists of clean samples and samples with simple data augmentation (flipping, cropping, etc.).
# The training method is normal training until it reaches a certain accuracy on a clean test set.
# The loss function used is cross-entropy loss.
# The goal is to give the classifier basic classification capabilities.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import copy


# ==========================================
# 1. Configuration parameters
# ==========================================
class Config:
    data_dir = '../../data/Logo-2K+/train_and_test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-4
    num_classes = 2338
    save_path = "../../models/initial_classifier.pth"


# ==========================================
# 2. Data augmentation and loading
# ==========================================
def get_dataloaders():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),

            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(Config.data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=Config.batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    return dataloaders, dataset_sizes


# ==========================================
# 3. Model building
# ==========================================
def get_model(num_classes):
    print("Loading the ConvNeXt-Tiny pre-trained model...")
    model = models.convnext_tiny(weights='DEFAULT')

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model.to(Config.device)


# ==========================================
# 4. Initial Training
# ==========================================
def train_baseline(model, dataloaders, dataset_sizes):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=0.05)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Start initial Training...")

    for epoch in range(Config.num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch}/{Config.num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), Config.save_path)
                print(f"Optimal accuracy has been detected; the model has been saved to {Config.save_path}")

        print()
        epoch_end = time.time()
        print(f'Training time for this epoch: {(epoch_end - epoch_start) // 60:.0f}m {(epoch_end - epoch_start) % 60:.0f}s')

    time_elapsed = time.time() - since
    print(f'Pre-training completed, time taken: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Optimal validation set accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# 5. main function
# ==========================================
if __name__ == '__main__':
    dataloaders, dataset_sizes = get_dataloaders()

    model = get_model(Config.num_classes)

    trained_model = train_baseline(model, dataloaders, dataset_sizes)

    print(f"Initial training complete. Weight file generated: {Config.save_path}")