from core.evaluator import Evaluator
from core.error_analysis import ErrorAnalyzer
from core.confidence import ConfidenceAnalyzer

class HealthReport:
    def __init__(self, y_true, y_pred, y_proba):
        self.evaluator = Evaluator(y_true, y_pred)
        self.error_analyzer = ErrorAnalyzer(y_true, y_pred)
        self.confidence_analyzer = ConfidenceAnalyzer(y_true, y_pred, y_proba)

    def generate(self):
        results = []

        # Check 1: Overall Accuracy
        accuracy = self.evaluator.accuracy()
        if accuracy < 0.5:
            status = "❌ Fail"
            message = f"Overall accuracy is very low ({accuracy:.1%}). The model may not be usable in its current state."
        else:
            status = "✅ Pass"
            message = f"Overall accuracy is {accuracy:.1%}."
        results.append({"Check": "Overall Accuracy", "Status": status, "Details": message})

            # Check 2: Macro Recall
        macro_recall = self.evaluator.recall()['macro']
        if macro_recall < 0.5:
            status = "❌ Fail"
            message = f"Macro recall is very low ({macro_recall:.1%}). Some classes are being badly neglected by the model."
        elif macro_recall < 0.7:
            status = "⚠️ Warning"
            message = f"Macro recall is moderate ({macro_recall:.1%}). Consider reviewing per-class performance."
        else:
            status = "✅ Pass"
            message = f"Macro recall is healthy ({macro_recall:.1%})."
        results.append({"Check": "Macro Recall", "Status": status, "Details": message})
        
            # Check 3: Worst-Performing Class
        error_rates = self.error_analyzer.error_rate_by_class()
        worst_class = error_rates.index[0]
        worst_rate = error_rates.iloc[0]
        if worst_rate > 0.8:
            status = "❌ Fail"
            message = f"Class {worst_class} has a critical error rate ({worst_rate:.1%}). Consider collecting more data or using class weighting for this class."
        elif worst_rate > 0.5:
            status = "⚠️ Warning"
            message = f"Class {worst_class} has a high error rate ({worst_rate:.1%}). This class needs attention."
        else:
            status = "✅ Pass"
            message = f"No class has an error rate above 50%. Worst class ({worst_class}) sits at {worst_rate:.1%}."
        results.append({"Check": "Worst-Performing Class", "Status": status, "Details": message})
        
            # Check 4: Confidently Wrong Rate
        cw_rate = self.confidence_analyzer.confidently_wrong_rate()
        if cw_rate > 0.2:
            status = "❌ Fail"
            message = f"When this model is wrong, it's confidently wrong {cw_rate:.1%} of the time. This model's probability outputs are unreliable — consider calibration (e.g., Platt scaling) or reducing model complexity."
        elif cw_rate > 0.05:
            status = "⚠️ Warning"
            message = f"Confidently wrong rate is {cw_rate:.1%}. Some overconfidence present; monitor before production use."
        else:
            status = "✅ Pass"
            message = f"Confidently wrong rate is low ({cw_rate:.1%}). This model's confidence scores appear well-calibrated."
        results.append({"Check": "Confidently Wrong Rate", "Status": status, "Details": message})
        return results
