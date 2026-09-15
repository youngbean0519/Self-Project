import pandas as pd

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    average_precision_score,
)

from utils.labels import LABELS

def apply_threshold(y_prob, threshold=0.5):
    return (y_prob >= threshold).astype(int)

def evaluate_multilabel(y_true, y_pred, y_prob=None):
    results = {}

    results["micro_f1"] = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    results["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    results["hamming_losss"] = hamming_loss(
        y_true,
        y_pred,
    )

    if y_prob is not None:
        results["pr_auc_macro"] = average_precision_score(
            y_true,
            y_prob,
            average="macro",
        )

        results["pr_auc_micro"] = average_precision_score(
            y_true,
            y_prob,
            average="micro",
        )
    
    return results

def get_label_wise_f1(y_true, y_pred):
    scores = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    return pd.DataFrame({
        "label_id": range(len(LABELS)),
        "label": LABELS,
        "f1": scores,
    })