import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataTransforms:
    def __init__(self, dataset: str, use_augmentation=True, use_stats=False):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.use_stats = use_stats

    def get_train_transform(self):
        if not self.use_augmentation:
            return self.get_test_transform()

        if self.dataset.lower() in ("fashionmnist", "fashion_mnist", "fmnist"):
            # Grayscale (1-channel) friendly augmentations.
            # ColorJitter's saturation/hue and RandomVerticalFlip are dropped:
            # they either error on 1-channel images or flip garments upside
            # down in a way that isn't a realistic augmentation for clothing.
            transforms_list = [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
            if self.use_stats:
                transforms_list.append(transforms.Normalize((0.2860,), (0.3530,)))
            return transforms.Compose(transforms_list)

        # Default / CIFAR-10 style path (kept from the original CNN pipeline)
        transforms_list = [
            transforms.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.3, hue=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor()
        ]

        if self.use_stats and self.dataset.lower() == 'cifar10':
            transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        return transforms.Compose(transforms_list)

    def get_test_transform(self):
        transforms_list = [
            transforms.ToTensor(),
        ]

        if self.use_stats and self.dataset.lower() in ("fashionmnist", "fashion_mnist", "fmnist"):
            transforms_list.append(transforms.Normalize((0.2860,), (0.3530,)))
        elif self.use_stats and self.dataset.lower() == 'cifar10':
            transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        return transforms.Compose(transforms_list)
