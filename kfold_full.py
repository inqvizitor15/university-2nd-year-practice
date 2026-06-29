import os
import numpy as np
import random

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import models

from sklearn.model_selection import StratifiedKFold


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# CONFIG
# -----------------------------

DATASET_PATH = "dataset/augmented"

CLASSES = ["1", "2", "5", "10"]
NUM_CLASSES = 4

NUM_EPOCHS = 40
BATCH_SIZE = 8
NUM_FOLDS = 5
LR = 1e-3

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# -----------------------------
# TRANSFORMS
# -----------------------------

train_transform = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------------
# DATA LOADING
# -----------------------------

image_paths = []
labels = []

for label, cls in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_PATH, cls)

    for file in os.listdir(class_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(class_dir, file))
            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total images:", len(image_paths))


# -----------------------------
# DATASET
# -----------------------------

class CoinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# -----------------------------
# K-FOLD
# -----------------------------

skf = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=42
)

fold_results = []


# -----------------------------
# TRAIN LOOP
# -----------------------------

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):

    print("\n" + "="*50)
    print(f"FOLD {fold+1}")
    print("="*50)

    # split data
    train_paths = image_paths[train_idx]
    train_labels = labels[train_idx]

    val_paths = image_paths[val_idx]
    val_labels = labels[val_idx]

    # datasets
    train_dataset = CoinDataset(train_paths, train_labels, train_transform)
    val_dataset = CoinDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    # freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # classifier
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    best_acc = 0.0

    # epochs
    for epoch in range(NUM_EPOCHS):

        # ---------------------
        # TRAIN
        # ---------------------
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()

        train_acc = correct / total

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)

                outputs = model(imgs)

                _, preds = torch.max(outputs, 1)

                val_total += lbls.size(0)
                val_correct += (preds == lbls).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # save best model per fold
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"coin_model_fold{fold+1}.pth")

    print(f"\nBest Fold {fold+1} Accuracy: {best_acc:.4f}")

    fold_results.append(best_acc)


# -----------------------------
# FINAL RESULT
# -----------------------------

mean_acc = sum(fold_results) / len(fold_results)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

for i, acc in enumerate(fold_results):
    print(f"Fold {i+1}: {acc:.4f}")

print(f"\nMean Accuracy: {mean_acc:.4f}")