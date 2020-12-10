import json
import math
import multiprocessing
import random
import sys
import time
from queue import Queue

import numpy
import operator

from models import DecisionForest, DecisionTree, GaussianProcess, KNeighbors,\
    LinearDiscriminantAnalysis, NaiveBayes, PassiveAggressive,\
    Ridge, StochasticGradient, SupportVectorMachine
from models.Model import Model
from models.instances import Instances


class Phenotype(object):

    def __init__(self, phenotype_id, committee, gen_length):
        # phenotype attributes
        self.__id = phenotype_id
        self.__committee = committee
        self.__genLength = gen_length
        self.__fitness = 0.0
        self.__normalizedFitness = 0.0
        self.__isClassificationFinished = False
        self.__genes = [False for x in range(self.gen_length)]
        # classifier attributes
        self.__classifiers = {}
        self.__scores = []
        self.__time = 0.0
        self.__predictions = []
        self.__trainedModels = []
        self.__model = Model()
        self.__inst = Instances()
        # prep phase
        with open("models/_models_list.json") as f:
            self.__classifiers = json.load(f)
        self.create_random_genes()

    @property
    def committee(self):
        return self.__committee

    @property
    def gen_length(self):
        return self.__genLength

    @property
    def is_classification_finished(self):
        return self.__isClassificationFinished

    @property
    def genes(self):
        return self.__genes

    @genes.setter
    def genes(self, value):
        if len(value) == self.gen_length:
            self.__genes = value

    @property
    def time(self):
        return self.__time

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        if value >= 0.0:
            self.__fitness = value

    @property
    def normalizedFitness(self):
        return self.__normalizedFitness

    @normalizedFitness.setter
    def normalizedFitness(self, value):
        if 0.0 <= value <= 1.0:
            self.__normalizedFitness = value

    def copy(self, donor):
        self.__committee = donor.committee
        self.__genLength = donor.gen_length
        self.__fitness = donor.fitness
        self.__normalizedFitness = donor.normalizedFitness
        self.__isClassificationFinished = donor.is_classification_finished
        self.__genes = donor.genes

    # generate 10 positive genes (classifiers)
    def create_random_genes(self):
        it = 0
        while it < self.committee:
            index = random.randint(0, self.gen_length - 1)
            if not self.genes[index]:
                self.genes[index] = True
                it += 1

    def classification_did_finish(self):
        return self.__isClassificationFinished

    # Take results, vote for answer and calc score, then penalize time
    def calc_fitness(self):
        committee_answers = []
        for i in range(len(self.__predictions[0])):
            tmp = {}
            for predicts in self.__predictions:
                if predicts[i] in tmp.keys():
                    tmp[predicts[i]] += 1
                else:
                    tmp[predicts[i]] = 1
            inverse = [(value, key) for key, value in tmp.items()]
            val = max(inverse)[1]
            committee_answers.append(val)
        self.__fitness = pow(self.__model.calcScore(predictions=committee_answers), 2)
        return self.__fitness

    # choose classifiers from list and execute
    # then calculate fitness of all
    def run(self):
        self.__scores.clear()
        self.__time = 0.0
        self.__predictions.clear()
        self.__trainedModels.clear()
        self.__isClassificationFinished = False

        for i, gen in enumerate(self.genes):
            if gen:
                # Different approach
                start_time = time.time()
                score, predictions = self.__model.runClassifier(self.__inst.trained_classifiers[i])
                elapsed_time = time.time() - start_time
                self.__scores.append(score)
                self.__time += elapsed_time
                self.__predictions.append(predictions)

        self.__isClassificationFinished = True
        return self.calc_fitness()

    # Just for testing
    def test(self):
        pass
