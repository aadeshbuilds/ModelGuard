import pandas as pd
from scipy.stats import ks_2samp
class DriftDetector:
    def __init__(self, reference_data, current_data):
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_drift(self, threshold=0.05):
        results = []
        common_columns = [col for col in self.reference_data.columns if col in self.current_data.columns]

        for col in common_columns:
            stat, p_value = ks_2samp(self.reference_data[col], self.current_data[col])
            drifted = p_value < threshold
            results.append({
                'feature': col,
                'p_value': p_value,
                'drifted': drifted
            })

        return pd.DataFrame(results)