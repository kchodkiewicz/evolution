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
    METRICS_METHOD = "accuracy_score"

    def __init__(self, dataset_path, col, metrics):
        self.dataset = pd.read_csv(dataset_path)  # TODO maybe move to main - dunno if creating classifier class will
                                                  # invoke this init
        self.METRICS_METHOD = metrics
        # TODO clean dataset

        X = self.dataset.drop(columns=col)
        y = self.dataset[col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def calcScore(self, predictions):
        if self.METRICS_METHOD == "auc":
            fpr, tpr, thresholds = roc_curve(self.y_test, predictions, pos_label=2)
            score = auc(fpr, tpr)
        elif self.METRICS_METHOD == "accuracy_score":
            score = accuracy_score(self.y_test, predictions)
        else:
            score = f1_score(self.y_test, predictions)
        return score

    # Not sure if good solution - some classifiers may need separate methods
    def runClassifier(self, model):
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        score = self.calcScore(predictions)

        return score, predictions
