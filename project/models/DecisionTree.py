import time
from models.Model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


class DecisionTree(Model):

    def default_decision_tree(self):
        print("DecisionTree")
        start_time = time.process_time()

        model = DecisionTreeClassifier()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'
        # joblib.dump(model, model_dump)  # something is no yes

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump
