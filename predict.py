"""
Usage:
    $ python predict_image.py --weights 'weights/best.pth' --image 'data/image.png'
"""

import argparse

import numpy as np
import torch
from torchvision.io import read_image

from model.custom import UNet


CLASS = {"bg": 0, "wall": 1, "patch": 2}
CLASSES_NUM = 3


def preprocess_image(image):
    """
    :arg image: Image tensor
    :return: Normalized and reshaped tensor
    """

    # Normalize
    image = image/255
    # Add extra dimension to match (batch, ch, height, width)
    image = torch.unsqueeze(image, dim=0)
    return image


def process_output(model_outs):
    """
    Output processing
    :arg model_outs: Raw model output (batch, class_num, height, width)
    :return: Predicted classes
    """

    # Predict probability scores across out channels
    probabilities = torch.softmax(model_outs, dim=1)
    predicted_mask = torch.argmax(probabilities, dim=1)
    np_predicted_mask = predicted_mask.cpu().detach().numpy()
    # Create a list of all predicted classes from the mask
    predicted_classes = np.unique(np_predicted_mask)
    return predicted_classes


def predict(model, raw_image, device='cpu'):
    """
    :arg model: Trained model
    :arg raw_image: Image tensor
    :arg device: cpu or cuda
    :return: True if mistake found else False
    """

    image = preprocess_image(raw_image)
    image = image.to(device)
    # Set to evaluation mode
    model.to(device).eval()

    with torch.no_grad():
        # Predict
        outs = model(image)

    predicted_classes = process_output(outs)
    detected = True if CLASS['patch'] in predicted_classes else False
    return detected


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pth', help='trained weights path')
    parser.add_argument('--image', type=str, required=True, help='image path')
    args = parser.parse_args()
    return args


def main(opt):
    # Read image
    img_path = opt.image
    raw_image = read_image(img_path)

    # Initialize model and load trained weights
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=CLASSES_NUM)
    model.load_state_dict(torch.load(opt.weights))

    # Predict
    detected = predict(model, raw_image, device)
    if detected:
        print(f"Patch identified in the wall painting!")
    else:
        print(f"No Patch identified in the wall painting")


if __name__ == "__main__":
    opts = parse_opts()
    main(opts)
