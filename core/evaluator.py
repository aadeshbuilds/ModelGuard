from sklearn.metrics import (
                        accuracy_score, 
                        precision_score, 
                        recall_score, 
                        f1_score,
                        confusion_matrix,
                        classification_report
                        )

class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self):
        macro = precision_score(self.y_true, self.y_pred, average='macro')
        weighted = precision_score(self.y_true, self.y_pred, average='weighted')
        return {'macro': macro, 'weighted': weighted}

    def recall(self):
        macro = recall_score(self.y_true, self.y_pred, average='macro')
        weighted = recall_score(self.y_true, self.y_pred, average='weighted')
        return {'macro': macro, 'weighted': weighted}

    def f1(self):
        macro = f1_score(self.y_true, self.y_pred, average='macro')
        weighted = f1_score(self.y_true, self.y_pred, average='weighted')
        return {'macro': macro, 'weighted': weighted}
    
    def confusion(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def per_class_report(self):
        return classification_report(self.y_true, self.y_pred, output_dict=True)

