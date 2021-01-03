# Program wide structure containing dataset, subsets and metrics method etc.
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve


class Model(object):
    dataset = None
    X_train = []
    X_test = []
    X_validate = []
    y_train = []
    y_test = []
    y_validate = []
    METRICS_METHOD = "f1_score"
    verbose = False
    RUN_ID = None
    TEST = False
    PRE_TRAIN = False

    def calcScore(self, predictions, **kwargs):
        def calc(y, predicts):
            if Model.METRICS_METHOD == "auc":
                fpr, tpr, thresholds = roc_curve(y, predicts, pos_label=2)
                score = auc(fpr, tpr)
            elif Model.METRICS_METHOD == "accuracy_score":
                score = accuracy_score(y, predicts)
            else:
                score = f1_score(y, predicts, average='micro')
            return score
        if kwargs['verify']:
            return calc(self.y_validate, predictions)
        else:
            return calc(self.y_test, predictions)
