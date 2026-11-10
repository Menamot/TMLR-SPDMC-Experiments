import os
from pathlib import Path
import argparse
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def load_cifar_batch(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    X = batch[b"data"]
    y = np.array(batch[b"labels"])

    X = X.reshape(-1, 3, 32, 32)
    X = X.transpose(0, 2, 3, 1)

    return X, y


def load_cifar10(data_dir):
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X_batch, y_batch = load_cifar_batch(batch_path)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_test, y_test = load_cifar_batch(os.path.join(data_dir, "test_batch"))

    return X_train, y_train, X_test, y_test


class CIFAR10ArrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class VGG16GAPExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            feats = model(images)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract VGG16 GAP features from CIFAR-10")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data_dir = str(args.data_dir)
    output_path = args.output
    if output_path.exists():
        parser.error("Output already exists; choose a new filename")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load raw CIFAR-10
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)
    print("Raw train shape:", X_train.shape, y_train.shape)
    print("Raw test shape :", X_test.shape, y_test.shape)

    # ImageNet preprocessing for VGG16
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CIFAR10ArrayDataset(X_train, y_train, transform=transform)
    test_dataset = CIFAR10ArrayDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build feature extractor
    model = VGG16GAPExtractor().to(device)

    # Extract features
    X_train_feat, y_train_out = extract_features(model, train_loader, device)
    X_test_feat, y_test_out = extract_features(model, test_loader, device)

    print("Feature train shape:", X_train_feat.shape)
    print("Feature test shape :", X_test_feat.shape)

    # Save dataset
    np.savez_compressed(
        output_path,
        X_train=X_train_feat,
        y_train=y_train_out,
        X_test=X_test_feat,
        y_test=y_test_out,
    )

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
