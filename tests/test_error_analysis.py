import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.error_analysis import ErrorAnalyzer

def test_misclassified():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    analyzer = ErrorAnalyzer(y_true, y_pred)
    errors = analyzer.misclassified()
    assert len(errors) == 1
    assert errors.iloc[0]['true_label'] == 1
    assert errors.iloc[0]['predicted_label'] == 2

def test_error_rate_by_class():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    analyzer = ErrorAnalyzer(y_true, y_pred)
    rates = analyzer.error_rate_by_class()
    assert rates[1] == 0.5
    assert rates[2] == 0.0

def test_most_confused_with():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    analyzer = ErrorAnalyzer(y_true, y_pred)
    confused = analyzer.most_confused_with()
    assert confused[1] == 2