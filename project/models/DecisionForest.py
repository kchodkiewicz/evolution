# returns f1 score / accuracy and time
from models.Model import Model


class DecisionForest(Model):

    def func(self):
        print("DecisionForest")
        x = self.dataset
        return 1, 2, [1, 2, 3]
