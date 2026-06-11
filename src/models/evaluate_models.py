from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(y_true: Any, y_pred: Any) -> dict:
    """
    Calcula métricas para classificação binária.

    Classe negativa:
    - Movie = 0

    Classe positiva:
    - TV Show = 1
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    """
    Exibe as métricas no terminal.
    """
    print(f"\nModelo: {model_name}")
    print(f"Acurácia: {metrics['accuracy']}")
    print(f"Sensibilidade: {metrics['sensitivity']}")
    print(f"Especificidade: {metrics['specificity']}")
    print(f"Precisão: {metrics['precision']}")
    print(f"F1-score: {metrics['f1_score']}")

    confusion = metrics["confusion_matrix"]

    print("Matriz de confusão:")
    print(f"TN: {confusion['tn']} | FP: {confusion['fp']}")
    print(f"FN: {confusion['fn']} | TP: {confusion['tp']}")