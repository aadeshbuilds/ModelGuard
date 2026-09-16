import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import Evaluator

def test_accuracy():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    evaluator = Evaluator(y_true, y_pred)
    assert evaluator.accuracy() == 0.75

def test_precision_returns_valid_dict():
    y_true = [1, 1, 1, 2]
    y_pred = [1, 1, 2, 2]
    evaluator = Evaluator(y_true, y_pred)
    result = evaluator.precision()
    assert 'macro' in result
    assert 'weighted' in result
    assert 0 <= result['macro'] <= 1
    assert 0 <= result['weighted'] <= 1

def test_confusion():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    evaluator = Evaluator(y_true, y_pred)
    cm = evaluator.confusion()
    assert cm.tolist() == [[1, 1], [0, 2]]

def test_per_class_report():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    evaluator = Evaluator(y_true, y_pred)
    report = evaluator.per_class_report()
    assert '1' in report
    assert '2' in report
    assert 'accuracy' in report