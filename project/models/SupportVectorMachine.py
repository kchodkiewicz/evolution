import time
from models.Model import Model
from sklearn.svm import SVC, LinearSVC


class SupportVectorMachine(Model):

    def default_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC()
        model.fit(self.X_train, self.y_train)
        model_dump = f'{time.process_time()}-default-svm.joblib'

        predictions = model.predict(self.X_test)

        score = self.calcScore(predictions)

        elapsed_time = time.process_time() - start_time
        # TODO save results in db or some shit

        return score, elapsed_time, predictions, model_dump

    def linear_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = LinearSVC()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-svm.joblib'

        elapsed_time = time.process_time() - start_time
        # TODO save results in db or some shit

        return score, elapsed_time, predictions, model_dump

