import numpy as np
import pandas as pd

class ErrorAnalyzer:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def misclassified(self):
        misclassified_indices = np.where(self.y_true != self.y_pred)[0]
        errors_df = pd.DataFrame({
            'index': misclassified_indices,
            'true_label': np.array(self.y_true)[misclassified_indices],
            'predicted_label': np.array(self.y_pred)[misclassified_indices]
        })
        return errors_df
    
    def error_rate_by_class(self):
        errors_df = self.misclassified()
        error_counts = errors_df['true_label'].value_counts()
        total_counts = pd.Series(self.y_true).value_counts()
        error_rate = error_counts / total_counts
        return error_rate.sort_values(ascending=False)