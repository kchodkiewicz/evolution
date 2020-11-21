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

    def calcScore(self, predictions):
        if Model.METRICS_METHOD == "auc":
            fpr, tpr, thresholds = roc_curve(Model.y_test, predictions, pos_label=2)
            score = auc(fpr, tpr)
        elif Model.METRICS_METHOD == "accuracy_score":
            score = accuracy_score(Model.y_test, predictions, average='micro')
        else:
            score = f1_score(Model.y_test, predictions, average='micro')
        return score

    # Not sure if good solution - some classifiers may need separate methods
    def runClassifier(self, model):
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        score = self.calcScore(predictions)

        return score, predictions
