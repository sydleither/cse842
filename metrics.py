from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_metrics(pred):
    preds, labels = pred
    preds = preds.argmax(1)
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average=None)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics_multilabel(pred):
    preds, labels = pred
    preds = np.array(preds) >= 0
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average=None)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics_binary(pred):
    preds, labels = pred
    preds = preds.argmax(1)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }