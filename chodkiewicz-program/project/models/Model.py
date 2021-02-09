# Program wide structure containing dataset, subsets and metrics method etc.
from sklearn.metrics import f1_score, accuracy_score


def calcScore(predictions, **kwargs):
    def calc(y, predicts):
        if Model.METRICS_METHOD == "accuracy_score":
            score = accuracy_score(y, predicts)
        else:
            score = f1_score(y, predicts, average='micro')
        return score

    if kwargs['verify']:
        return calc(Model.y_validate, predictions)
    else:
        return calc(Model.y_test, predictions)


class Model(object):
    dataset = None
    X_train = []
    X_test = []
    X_validate = []
    y_train = []
    y_test = []
    y_validate = []
    METRICS_METHOD = "f1_score"
    VERBOSE = False
    RUN_ID = None
    TEST = False
    PRE_TRAIN = False
