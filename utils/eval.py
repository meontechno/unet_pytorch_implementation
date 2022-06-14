"""
Model evaluation - Computing Intersection over Union (IoU) metric
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(cm):
    """
    Measure the number of pixels common between the target and prediction masks
    divided by the total number of pixels present across both masks
    :arg cm: Confusion matrix (predicted masks, ground truth masks)
    :returns: Class IoU probabilities and mean IoU
    """
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)
    denominator = sum_over_row + sum_over_col - true_positives
    iou = true_positives / denominator
    return iou, np.nanmean(iou)


def model_eval(model, val_loader, n_classes, device='cuda'):
    """
    :arg model: model
    :arg val_loader: Validation data loader
    :arg n_classes: Total classes
    :arg device: cuda or cpu
    :returns: Class IoU probabilities and mean IoU
    """
    model.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes, n_classes))

    for i, (images, gt_masks) in enumerate(val_loader):
        images = images.to(device)
        gt_masks = gt_masks.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for j in range(len(gt_masks)):
            true = gt_masks[j].cpu().detach().numpy().flatten()
            pred = preds[j].cpu().detach().numpy().flatten()
            cm += confusion_matrix(true, pred, labels=labels)

    class_iou, mean_iou = compute_iou(cm)
    return class_iou, mean_iou
