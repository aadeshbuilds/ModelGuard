import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.confidence import ConfidenceAnalyzer

def test_confidence():
    y_true = [1, 1, 2]
    y_pred = [1, 2, 2]
    y_proba = [
        [0.9, 0.1],
        [0.3, 0.7],
        [0.2, 0.8],
    ]
    analyzer = ConfidenceAnalyzer(y_true, y_pred, y_proba)
    conf = analyzer.confidence()
    assert list(conf) == [0.9, 0.7, 0.8]
    
def test_confidently_wrong():
    y_true = [1, 1, 2]
    y_pred = [1, 2, 2]
    y_proba = [
        [0.9, 0.1],
        [0.3, 0.7],
        [0.2, 0.8],
    ]
    analyzer = ConfidenceAnalyzer(y_true, y_pred, y_proba)
    cw = analyzer.confidently_wrong(threshold=0.6)
    assert len(cw) == 1
    assert cw.iloc[0]['true_label'] == 1
    assert cw.iloc[0]['predicted_label'] == 2

def test_confidently_wrong_rate():
    y_true = [1, 1, 2]
    y_pred = [1, 2, 2]
    y_proba = [
        [0.9, 0.1],
        [0.3, 0.7],
        [0.2, 0.8],
    ]
    analyzer = ConfidenceAnalyzer(y_true, y_pred, y_proba)
    rate = analyzer.confidently_wrong_rate(threshold=0.6)
    assert rate == 1.0

def test_confidently_wrong_rate_perfect_model():
    y_true = [1, 2]
    y_pred = [1, 2]
    y_proba = [[0.9, 0.1], [0.1, 0.9]]
    analyzer = ConfidenceAnalyzer(y_true, y_pred, y_proba)
    rate = analyzer.confidently_wrong_rate()
    assert rate == 0.0