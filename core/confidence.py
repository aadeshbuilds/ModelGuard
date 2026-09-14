import numpy as np
import pandas as pd

class ConfidenceAnalyzer:
    def __init__(self, y_true, y_pred, y_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def confidence(self):
        return np.max(self.y_proba, axis=1)

    def confidently_wrong(self, threshold=0.9):
        confidence = self.confidence()
        wrong = np.array(self.y_true) != np.array(self.y_pred)
        high_confidence = confidence > threshold
        mask = wrong & high_confidence
        indices = np.where(mask)[0]
        return pd.DataFrame({
            'index': indices,
            'true_label': np.array(self.y_true)[indices],
            'predicted_label': np.array(self.y_pred)[indices],
            'confidence': confidence[indices]
        })

    def confidently_wrong_rate(self, threshold=0.9):
        n_wrong = (np.array(self.y_true) != np.array(self.y_pred)).sum()
        n_confidently_wrong = self.confidently_wrong(threshold).shape[0]
        return n_confidently_wrong / n_wrong