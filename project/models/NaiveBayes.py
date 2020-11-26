import time
from models.Model import Model
from sklearn.naive_bayes import GaussianNB, MultinomialNB


class NaiveBayes(Model):

    def gaussian_nb(self):
        start_time = time.process_time()

        model = GaussianNB()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_nb_smoothing1(self):
        start_time = time.process_time()

        model = GaussianNB(var_smoothing=1)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_nb_smoothing0(self):
        start_time = time.process_time()

        model = GaussianNB(var_smoothing=0)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def gaussian_nb_smoothingMax(self):
        start_time = time.process_time()

        model = GaussianNB(var_smoothing=1e9)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb(self):
        start_time = time.process_time()

        model = MultinomialNB()
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb_alpha0(self):
        start_time = time.process_time()

        model = MultinomialNB(alpha=1e-10)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb_alpha10(self):
        start_time = time.process_time()

        model = MultinomialNB(alpha=1e10)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb_fitpriorFalse(self):
        start_time = time.process_time()

        model = MultinomialNB(fit_prior=False)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb_alpha0_fitpriorFalse(self):
        start_time = time.process_time()

        model = MultinomialNB(alpha=1e-10, fit_prior=False)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump

    def multinomial_nb_alpha10_fitpriorFalse(self):
        start_time = time.process_time()

        model = MultinomialNB(alpha=1e10, fit_prior=False)
        score, predictions = self.runClassifier(model)
        model_dump = f'{time.process_time()}-naive-bayes.joblib'

        elapsed_time = time.process_time() - start_time
        return score, elapsed_time, predictions, model_dump
