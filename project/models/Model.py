import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve


# Handler for file opening, cleaning data, splitting dataset etc.
class Model(object):
    dataset = None
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    METRICS_METHOD = "f1_score"

    classifiers_instances = []
    __classifiers = {}

    def trainClassifiers(self):
        # prep phase
        with open("models/_models_list.json") as f:
            self.__classifiers = json.load(f)

        for i in range(60):
            s = str(i + 10)
            digit1st = s[0]
            digit2nd = s[1]
            # Get filename, classname and method name from json file and execute
            classifier = self.__classifiers[digit1st]["version"][digit2nd]
            filename = getattr(sys.modules[__name__], self.__classifiers[digit1st]["name"])
            classname = getattr(filename, self.__classifiers[digit1st]["name"])
            class_object = classname()
            method = getattr(class_object, classifier, lambda: "Invalid classifier")
            score, time, predictions, model_dump = method()

            self.classifiers_instances.append()

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
