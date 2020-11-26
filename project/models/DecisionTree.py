import time
from models.Model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Model):

    def decision_tree(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_criterionEntropy(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(criterion='entropy')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_splitterRandom(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(splitter='random')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_splitterRandom_criterionEntropy(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(splitter='random', criterion='entropy')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_maxfeaturesSqrt(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(max_features='sqrt')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_maxfeaturesLog2(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(max_features='log2')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_criterionEntropy_maxfeaturesLog2(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(criterion='entropy', max_features='log2')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_splitterRandom_maxfeaturesLog2(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(splitter='random', max_features='log2')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_criterionEntropy_maxfeaturesSqrt(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(criterion='entropy', max_features='sqrt')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def decision_tree_splitterRandom_maxfeaturesSqrt(self):
        start_time = time.process_time()

        model = DecisionTreeClassifier(splitter='random', max_features='sqrt')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-decision-tree.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump
