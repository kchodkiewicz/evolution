# returns f1 score / accuracy and time
from models import Model


class DecisionForest(Model.Model):

    def func(self):
        print("DecisionForest")
        x = self.dataset
        return 1, 2, [1, 2, 3]
