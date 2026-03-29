import copy

import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

IMAGE_MEAN = [0.7896, 0.6630, 0.6340]
IMAGE_STD = [0.2228, 0.3200, 0.3320]

# Shared post-processing: ToTensor + Normalize (used by all transforms)
_post = [T.ToTensor(), T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)]


def get_train_transform():
    return T.Compose([T.RandomResizedCrop(224), *_post])


def get_val_transform():
    return T.Compose([T.CenterCrop(224), *_post])


# Same as val — images are already 256x256 from resize_dataset.py
get_inference_transform = get_val_transform


def load_datasets(root_dir: str, test_size=0.3, random_state=42):
    full_dataset = ImageFolder(root=root_dir)
    labels = [label for _, label in full_dataset.samples]

    # Stratified split — keeps the same class ratio in each split
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)), test_size=test_size, stratify=labels, random_state=random_state
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=random_state
    )

    # Each subset gets its own copy so transforms don't conflict
    train_data = Subset(copy.deepcopy(full_dataset), train_idx)
    train_data.dataset.transform = get_train_transform()
    val_data = Subset(copy.deepcopy(full_dataset), val_idx)
    val_data.dataset.transform = get_val_transform()
    test_data = Subset(copy.deepcopy(full_dataset), test_idx)
    test_data.dataset.transform = get_val_transform()

    return train_data, val_data, test_data, full_dataset.classes


def get_loaders(root_dir: str, batch_size: int = 32):
    train_data, val_data, test_data, classes = load_datasets(root_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, classes
