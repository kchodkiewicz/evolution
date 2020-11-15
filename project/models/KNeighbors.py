import time
from models.Model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNeighbors(Model):

    def k_neighbors(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_neighbors2(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(n_neighbors=2)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_neighbors10(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(n_neighbors=10)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_weightsDistance(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(weights='distance')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoBallTree(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='ball_tree')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoKDTree(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='kd_tree')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoBrute(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='brute')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoBallTree_weightsDistance(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoKDTree_weightsDistance(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='kd_tree', weights='distance')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def k_neighbors_algoBrute_weightsDistance(self):
        print("KNeighbors")
        start_time = time.process_time()

        model = KNeighborsClassifier(algorithm='brute', weights='distance')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-k-neighbors.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

