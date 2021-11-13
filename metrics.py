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


def compute_metrics_token(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=2)

    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(len(labels)):
        if (2 in labels[i] and 2 in preds[i]) or (2 in labels[i] and 1 in preds[i]):
            tp += 1
        elif (2 in labels[i] and 2 not in preds[i]) or (2 in labels[i] and 1 not in preds[i]):
            fn += 1
        elif (2 not in labels[i] and 2 in preds[i]) or (2 not in labels[i] and 1 in preds[i]):
            fp += 1
        elif (2 not in labels[i] and 2 not in preds[i]) or (2 not in labels[i] and 1 not in preds[i]):
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }