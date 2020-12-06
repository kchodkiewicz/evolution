import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
from models.instances import Instances


# Handler for file opening, cleaning data, splitting dataset etc.
class Model(Instances):
    dataset = None
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    METRICS_METHOD = "f1_score"

    def calcScore(self, predictions):
        if Model.METRICS_METHOD == "auc":
            fpr, tpr, thresholds = roc_curve(self.y_test, predictions, pos_label=2)
            score = auc(fpr, tpr)
        elif Model.METRICS_METHOD == "accuracy_score":
            score = accuracy_score(self.y_test, predictions)
        else:
            score = f1_score(self.y_test, predictions, average='micro')
        return score

    def runClassifier(self, model):
        #  model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        score = self.calcScore(predictions)

        return score, predictions
