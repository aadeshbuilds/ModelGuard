import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.comparison import ModelComparator

def test_compare_accuracy_winner():
    y_true = [1, 1, 2, 2]
    y_pred_a = [1, 2, 2, 2]
    y_pred_b = [1, 1, 2, 2]
    y_proba_a = [[0.8,0.2],[0.3,0.7],[0.2,0.8],[0.1,0.9]]
    y_proba_b = [[0.9,0.1],[0.85,0.15],[0.1,0.9],[0.05,0.95]]

    comparator = ModelComparator(y_true, y_pred_a, y_proba_a, y_pred_b, y_proba_b, name_a="A", name_b="B")
    result = comparator.compare()

    assert result.loc['Accuracy', 'A'] == 0.75
    assert result.loc['Accuracy', 'B'] == 1.0
    assert result.loc['Accuracy', 'Better Model'] == 'B'