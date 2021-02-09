# Program wide structure containing dataset, subsets, metrics method etc.

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
