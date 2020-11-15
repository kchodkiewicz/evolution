import time
from models.Model import Model
from sklearn.gaussian_process import GaussianProcessClassifier


class GaussianProcess(Model):

    def gaussian_process(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_warmstartTrue(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(warm_start=True)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_restarts1(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(n_restarts_optimizer=1)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_restarts1_warmstartTrue(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(n_restarts_optimizer=1, warm_start=True)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_iterpredict1000(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(max_iter_predict=1000)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_warmstartTrue_iterpredict1000(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(warm_start=True, max_iter_predict=1000)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_restarts1_iterpredict1000(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(n_restarts_optimizer=1, max_iter_predict=1000)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_restarts1_warmstartTrue_iterpredict1000(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(n_restarts_optimizer=1, warm_start=True, max_iter_predict=1000)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_iterpredict10(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(max_iter_predict=10)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_process_restarts5(self):
        print("GaussianProcess")
        start_time = time.process_time()

        model = GaussianProcessClassifier(n_restarts_optimizer=5)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-gaussian-process.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump
