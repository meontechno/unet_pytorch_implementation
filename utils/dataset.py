"""
Custom dataset class and dataset utils
"""

import os
from glob import glob

import numpy
import natsort
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split

COLOR_ENC = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}


def rgb_to_flat_mask(mask):
    """
    Converting rgb mask into class encoded flat mask
    :arg mask: rgb mask
    :return: flat mask with class encoding
    """
    mask_np = numpy.array(mask.permute(1, 2, 0))
    temp_mask = numpy.zeros((mask_np.shape[0], mask_np.shape[1]))
    for cls, color in COLOR_ENC.items():
        temp_mask[numpy.all(mask_np == color, axis=2)] = cls
    return torch.from_numpy(temp_mask).long()


class RoomDataset(Dataset):
    """
    Training dataset
    :arg img_dir: Path to all training images
    :arg mask_dir: Path to all training masks
    """
    def __init__(self, img_dir, mask_dir):
        self.images = natsort.natsorted(glob(os.path.join(img_dir, '*.png')))
        self.masks = natsort.natsorted(glob(os.path.join(mask_dir, '*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Read image and normalize
        image = read_image(self.images[idx])/255
        mask = rgb_to_flat_mask(read_image(self.masks[idx]))
        return image, mask


def create_train_val_loaders(images_path, masks_path, batch):
    """
    Split the dataset into train and val sets according to split ratio
    90% to train and 10% to val
    :arg images_path: Path to images
    :arg masks_path: Path to masks
    :arg batch: Batch size
    :return: train and val data loaders
    """
    dataset = RoomDataset(images_path, masks_path)
    # Split dataset
    dataset_len = len(dataset)
    val_len = int(0.1 * dataset_len)
    lens = [dataset_len - val_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lens)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)
    return train_loader, val_loader


def test():
    """
    Sample test to verify image and corresponding mask have right shapes
    """
    dataset = RoomDataset('data/train/images', 'data/train/masks')
    image, mask = dataset.__getitem__(0)
    print(f"Image:\n {image} \nMask:\n {mask}")
    print(f"Image shape: {image.shape} \nMask shape: {mask.shape}")
    assert image.shape[1:] == mask.shape


if __name__ == "__main__":
    test()
