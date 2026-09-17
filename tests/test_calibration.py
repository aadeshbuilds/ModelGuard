import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import Calibration

def test_brier_score_perfect_model():
    y_true = [1, 2]
    y_pred = [1, 2]
    y_proba = [
        [1.0, 0.0],  # 100% confident, correct
        [0.0, 1.0],  # 100% confident, correct
    ]
    calibration = Calibration(y_true, y_pred, y_proba)
    score = calibration.brier_score()
    assert score == 0.0

def test_calibration_summary_runs_with_list_input():
    y_true = [1, 2, 1, 2]
    y_pred = [1, 2, 2, 2]
    y_proba = [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9],
    ]
    calibration = Calibration(y_true, y_pred, y_proba)
    summary = calibration.calibration_summary()
    assert len(summary) > 0