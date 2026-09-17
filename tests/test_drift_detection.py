import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core.drift_detection import DriftDetector

def test_detects_drift_in_shifted_feature():
    reference = pd.DataFrame({'feature_a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    current = pd.DataFrame({'feature_a': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]})

    detector = DriftDetector(reference, current)
    results = detector.detect_drift()

    assert results.iloc[0]['drifted'] == True
    
def test_no_drift_on_identical_distributions():
    reference = pd.DataFrame({'feature_a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    current = pd.DataFrame({'feature_a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    detector = DriftDetector(reference, current)
    results = detector.detect_drift()

    assert results.iloc[0]['drifted'] == False