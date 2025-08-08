
# utils/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_true, y_pred, average='weighted'), 4),
        "f1_score": round(f1_score(y_true, y_pred, average='weighted'), 4),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
    }
