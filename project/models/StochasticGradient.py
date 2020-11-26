import time
from models.Model import Model
from sklearn.linear_model import SGDClassifier


class StochasticGradient(Model):

    def stochastic_gradient(self):
        start_time = time.process_time()

        model = SGDClassifier()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossLog(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='log')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossModifiedHumber(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='modified_huber')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossSquaredHinge(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='squared_hinge')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossPerception(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='perceptron')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_penaltyElasticNet(self):
        start_time = time.process_time()

        model = SGDClassifier(penalty='elasticnet')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossLog_penaltyElasticNet(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='log', penalty='elasticnet')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossModifiedHumber_penaltyElasticNet(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='modified_huber', penalty='elasticnet')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossSquaredHinge_penaltyElasticNet(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='squared_hinge', penalty='elasticnet')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def stochastic_gradient_lossPerception_penaltyElasticNet(self):
        start_time = time.process_time()

        model = SGDClassifier(loss='perceptron', penalty='elasticnet')
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-stochastic-gradient.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump
