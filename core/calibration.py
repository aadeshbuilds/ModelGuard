import numpy as np
import pandas as pd

class Calibration:
    def __init__(self, y_true, y_pred, y_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def calibration_summary(self, n_bins=10):
        confidence = np.max(self.y_proba, axis=1)
        correct = (np.array(self.y_true) == np.array(self.y_pred))
        bins = np.linspace(0, 1, n_bins + 1)
        bin_labels = pd.cut(confidence, bins=bins, include_lowest=True)
        df = pd.DataFrame({'confidence': confidence, 'correct': correct, 'bin': bin_labels})
        summary = df.groupby('bin', observed=True).agg(
            avg_confidence=('confidence', 'mean'),
            actual_accuracy=('correct', 'mean'),
            count=('correct', 'size')
        )
        return summary

    def brier_score(self):
        y_true_arr = np.array(self.y_true)
        y_proba_arr = np.array(self.y_proba)
        n_classes = y_proba_arr.shape[1]
        classes = sorted(set(y_true_arr) | set(np.array(self.y_pred)))
        if len(classes) < n_classes:
            # y_proba has columns for classes never observed in this sample —
            # fall back to a plain numeric class range so column order still lines up
            classes = list(range(1, n_classes + 1)) if min(y_true_arr) >= 1 else list(range(n_classes))
        y_true_onehot = np.zeros((len(y_true_arr), n_classes))
        for i, c in enumerate(classes):
            y_true_onehot[:, i] = (y_true_arr == c).astype(int)
        return np.mean(np.sum((y_proba_arr - y_true_onehot) ** 2, axis=1))
