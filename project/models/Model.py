import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, classification_report


# Handler for file opening, cleaning data, splitting dataset etc.
class Model(object):
    dataset = None
    X_train = []
    X_test = []
    X_validate = []
    y_train = []
    y_test = []
    y_validate = []
    METRICS_METHOD = "f1_score"

    def calcScore(self, predictions, **kwargs):
        if kwargs['verify']:
            if Model.METRICS_METHOD == "auc":
                fpr, tpr, thresholds = roc_curve(self.y_validate, predictions, pos_label=2)
                score = auc(fpr, tpr)
            elif Model.METRICS_METHOD == "accuracy_score":
                score = accuracy_score(self.y_validate, predictions)
            else:
                score = f1_score(self.y_validate, predictions, average='micro')
            return score
        else:
            if Model.METRICS_METHOD == "auc":
                fpr, tpr, thresholds = roc_curve(self.y_test, predictions, pos_label=2)
                score = auc(fpr, tpr)
            elif Model.METRICS_METHOD == "accuracy_score":
                score = accuracy_score(self.y_test, predictions)
            else:
                score = f1_score(self.y_test, predictions, average='micro')
            return score

    def runClassifier(self, model):
        predictions = model.predict(self.X_test)
        score = self.calcScore(predictions)
        return score, predictions
