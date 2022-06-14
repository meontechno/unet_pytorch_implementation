"""
Model training on a custom dataset

Usage:
    $ python train.py --data 'data/train' --batch 8 --epochs 100
"""

import argparse

import torch
from tqdm import tqdm

from model.custom import UNet
from utils.dataset import create_train_val_loaders
from utils.eval import model_eval

CLASSES_NUM = 3


def train(model, device, train_loader, val_loader, epochs=10, optimizer=None, loss_fn=None):
    best_mean_iou = 0
    for epoch in range(epochs):
        # Set to training mode
        model.train()

        epoch_loss = 0
        # Using tqdm for displaying progress bar during each epoch
        loader = tqdm(train_loader)

        for i, (images, gt_masks) in enumerate(loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            # Predict
            outs = model(images)

            # Compute loss
            loss = loss_fn(outs, gt_masks)
            epoch_loss += loss.item()

            # Clear previous gradients
            optimizer.zero_grad()
            # Automatic backpropagation
            loss.backward()
            # Optimizer update
            optimizer.step()

            loader.set_postfix(epoch=epoch+1, loss=loss.item())

        print(f"Epoch {epoch} : Loss {epoch_loss / i:.2f}")
        # Model evaluation (Compute IoU metric)
        class_iou, mean_iou = model_eval(model, val_loader, CLASSES_NUM, device)
        print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}')

        # Save model after every epoch
        state_dict = model.state_dict()
        torch.save(state_dict, "weights/last.pth")
        # Save model with best accuracy
        if mean_iou > best_mean_iou:
            torch.save(state_dict, "weights/best.pth")
            best_mean_iou = mean_iou


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/images_train', help='training data path')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    return args


def main(opt):
    batch = opt.batch
    epochs = opt.epochs
    images_path = opt.data + '/images'
    masks_path = opt.data + '/masks'

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = UNet(in_ch=3, out_ch=CLASSES_NUM)
    model.to(device)

    # Create train and val loaders
    train_loader, val_loader = create_train_val_loaders(images_path, masks_path, batch)

    # Gradient optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Loss function
    # (Dice loss can also be used instead of cross entropy loss)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    train(model, device, train_loader, val_loader, epochs=epochs, optimizer=optimizer, loss_fn=loss_fn)


if __name__ == "__main__":
    opts = parse_opts()
    main(opts)
