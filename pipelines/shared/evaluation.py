from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_model(name, estimator, X_test, y_test, label_names):
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )[2]
    cm = confusion_matrix(y_test, y_pred)

    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)

    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr")
        except Exception:
            roc_auc = None

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
        },
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr": roc_auc,
        "classification_report": report,
    }
    return metrics
