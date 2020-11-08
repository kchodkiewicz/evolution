import time
from models.Model import Model
from sklearn.tree import DecisionTreeClassifier


# returns f1 score / accuracy and time
class DecisionTree(Model):

    def func(self):
        print("DecisionTree")
        start_time = time.process_time()

        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        score = self.calcScore(predictions)

        elapsed_time = time.process_time() - start_time
        # TODO save results in db or some shit

        return score, elapsed_time, [1, 2, 3]
