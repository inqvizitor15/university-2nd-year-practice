import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision import models

from sklearn.model_selection import StratifiedKFold


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "dataset/augmented"

CLASSES = ["1", "2", "5", "10"]
NUM_CLASSES = 4

NUM_FOLDS = 3
NUM_EPOCHS = 10
BATCH_SIZE = 12
LR = 1e-3

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# TRANSFORMS (БЕЗ АУГМЕНТАЦИИ НА ЛЕТУ)
# =========================================================

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = train_transform


# =========================================================
# DATASET
# =========================================================

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


# =========================================================
# LOAD DATASET
# =========================================================

image_paths = []
labels = []

for label, cls in enumerate(CLASSES):

    class_dir = os.path.join(DATASET_PATH, cls)

    for file in os.listdir(class_dir):

        if file.lower().endswith((".jpg", ".png", ".jpeg")):

            image_paths.append(os.path.join(class_dir, file))
            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total images:", len(image_paths))


# =========================================================
# MODEL FACTORY
# =========================================================

def create_model():

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    # freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # UNFREEZE LAST BLOCKS (ВАЖНОЕ УЛУЧШЕНИЕ)
    for block in model.features[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # classifier
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    return model


# =========================================================
# OPTIMIZER
# =========================================================

def create_optimizer(model):

    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# K-FOLD
# =========================================================

skf = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=SEED
)

fold_results = []


# =========================================================
# START K-FOLD TRAINING
# =========================================================

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):

    print("\n" + "=" * 60)
    print(f"FOLD {fold + 1}")
    print("=" * 60)

    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------

    train_paths = image_paths[train_idx]
    train_labels = labels[train_idx]

    val_paths = image_paths[val_idx]
    val_labels = labels[val_idx]

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    train_dataset = CoinDataset(train_paths, train_labels, train_transform)
    val_dataset = CoinDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = create_model().to(device)

    optimizer = create_optimizer(model)

    best_acc = 0.0

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(NUM_EPOCHS):

        # -------------------------
        # TRAIN
        # -------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, lbls in train_loader:

            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            train_total += lbls.size(0)
            train_correct += (preds == lbls).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for imgs, lbls in val_loader:

                imgs = imgs.to(device)
                lbls = lbls.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                val_total += lbls.size(0)
                val_correct += (preds == lbls).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # -------------------------
        # PRINT
        # -------------------------

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # -------------------------
        # SAVE BEST MODEL PER FOLD
        # -------------------------

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                f"coin_model_fold{fold+1}.pth"
            )

    # -----------------------------------------------------
    # END FOLD
    # -----------------------------------------------------

    print(f"\nBest Fold {fold+1} Accuracy: {best_acc:.4f}")

    fold_results.append(best_acc)


# =========================================================
# FINAL RESULTS
# =========================================================

mean_acc = sum(fold_results) / len(fold_results)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for i, acc in enumerate(fold_results):
    print(f"Fold {i+1}: {acc:.4f}")

print(f"\nMean Accuracy: {mean_acc:.4f}")