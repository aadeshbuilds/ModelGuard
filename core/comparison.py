from core.evaluator import Evaluator
from core.confidence import ConfidenceAnalyzer
import pandas as pd


class ModelComparator:
    def __init__(self, y_true, y_pred_a, y_proba_a, y_pred_b, y_proba_b, name_a="Model A", name_b="Model B"):
        self.evaluator_a = Evaluator(y_true, y_pred_a)
        self.evaluator_b = Evaluator(y_true, y_pred_b)
        self.confidence_a = ConfidenceAnalyzer(y_true, y_pred_a, y_proba_a)
        self.confidence_b = ConfidenceAnalyzer(y_true, y_pred_b, y_proba_b)
        self.name_a = name_a
        self.name_b = name_b

    def compare(self):
        higher_is_better = {
            'Accuracy': True,
            'Macro F1': True,
            'Macro Recall': True,
            'Confidently Wrong Rate': False
        }

        metrics = {
            'Accuracy': (self.evaluator_a.accuracy(), self.evaluator_b.accuracy()),
            'Macro F1': (self.evaluator_a.f1()['macro'], self.evaluator_b.f1()['macro']),
            'Macro Recall': (self.evaluator_a.recall()['macro'], self.evaluator_b.recall()['macro']),
            'Confidently Wrong Rate': (self.confidence_a.confidently_wrong_rate(), self.confidence_b.confidently_wrong_rate())
        }

        df = pd.DataFrame(metrics, index=[self.name_a, self.name_b]).T
        df['Difference'] = df[self.name_b] - df[self.name_a]

        def determine_winner(row, metric_name):
            if higher_is_better[metric_name]:
                return self.name_a if row[self.name_a] > row[self.name_b] else self.name_b
            else:
                return self.name_a if row[self.name_a] < row[self.name_b] else self.name_b

        df['Better Model'] = [determine_winner(df.loc[m], m) for m in df.index]
        return df