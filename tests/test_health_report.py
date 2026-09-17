import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.health_report import HealthReport

def test_health_report_flags_bad_model():
    y_true = [1, 1, 1, 1, 2]
    y_pred = [2, 2, 2, 2, 1]
    y_proba = [[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.9,0.1]]

    health = HealthReport(y_true, y_pred, y_proba)
    checks = health.generate()

    accuracy_check = next(c for c in checks if c['Check'] == 'Overall Accuracy')
    assert accuracy_check['Status'] == '❌ Fail'

def test_health_report_passes_good_model():
    y_true = [1, 1, 2, 2, 1, 2]
    y_pred = [1, 1, 2, 2, 1, 2]  # perfect model
    y_proba = [[0.95,0.05],[0.9,0.1],[0.05,0.95],[0.1,0.9],[0.92,0.08],[0.05,0.95]]

    health = HealthReport(y_true, y_pred, y_proba)
    checks = health.generate()

    accuracy_check = next(c for c in checks if c['Check'] == 'Overall Accuracy')
    assert accuracy_check['Status'] == '✅ Pass'

    confidently_wrong_check = next(c for c in checks if c['Check'] == 'Confidently Wrong Rate')
    assert confidently_wrong_check['Status'] == '✅ Pass'