import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def evaluate_model(model, X_test, y_test, class_names):

    predictions = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(class_names)))
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "report_text": report_text,
        "predictions": y_pred
    }
