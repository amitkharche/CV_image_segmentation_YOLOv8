import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0

def compute_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return (2. * intersection) / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0

def evaluate_masks(pred_mask, true_mask):
    iou = compute_iou(pred_mask, true_mask)
    dice = compute_dice(pred_mask, true_mask)
    flat_pred = pred_mask.flatten()
    flat_true = true_mask.flatten()
    precision = precision_score(flat_true, flat_pred, zero_division=0)
    recall = recall_score(flat_true, flat_pred, zero_division=0)
    f1 = f1_score(flat_true, flat_pred, zero_division=0)
    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
