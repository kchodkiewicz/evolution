import time
from models.Model import Model
from sklearn.svm import SVC, LinearSVC


class SupportVectorMachine(Model):

    def default_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-default-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def linear_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = LinearSVC()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_rbf_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='rbf')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_rbf_svm_cMIN_gMIN(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='rbf', C=1e-2, gamma=1e-1)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_rbf_svm_cMAX_gMAX(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='rbf', C=1e2, gamma=1e1)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_polynomial_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='poly')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_polynomial_svm_d1(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='poly', degree=1)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump

    def kernel_sigmoid_svm(self):
        print("SupportVectorMachine")
        start_time = time.process_time()

        model = SVC(kernel='sigmoid')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-linear-svm.joblib'

        elapsed_time = time.process_time() - start_time

        return score, elapsed_time, predictions, model_dump