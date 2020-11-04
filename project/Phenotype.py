import random


class Phenotype:

    def __init__(self, committee, gen_length):
        # phenotype attributes
        self.__committee = committee
        self.__genLength = gen_length
        self.__fitness = 0.0
        self.__normalizedFitness = 0.0
        self.__isBest = False
        self.__isClassificationFinished = False
        self.__genes = [False for x in range(self.gen_length)]
        # classifier attributes

        self.create_random_genes()

    @property
    def committee(self):
        return self.__committee

    @property
    def gen_length(self):
        return self.__genLength

    @property
    def is_best(self):
        return self.__isBest

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

    def calc_fitness(self):
        pass

    def run(self):
        self.__isClassificationFinished = False
        for gen in self.genes:
            if gen:
                # choose classifiers from list and execute

                pass
        self.calc_fitness()
        self.__isClassificationFinished = True
