import random


class Phenotype:

    def __init__(self, committee, gen_length):
        self.committee = committee
        self.genLength = gen_length
        self.fitness = 0.0
        self.normalizedFitness = 0.0
        self.isBest = False
        self.isClassificationFinished = False
        self.genes = [False for x in range(self.genLength)]
        self.create_random_genes()

    # generate 10 positive genes (classifiers)
    def create_random_genes(self):
        it = 0
        while it < self.committee:
            index = random.randint(0, self.genLength - 1)
            if not self.genes[index]:
                self.genes[index] = True
                it += 1

    def copy(self, parent):
        # TODO fix dat shit
        self.committee = parent.committee
        self.genLength = parent.genesLength
        self.genes = parent.get_genes()

    @property
    def genes(self):
        return self.genes

    @genes.setter
    def genes(self, value):
        if len(value) == self.genLength:
            self.genes = value

    @property
    def fitness(self):
        return self.fitness

    @fitness.setter
    def fitness(self, value):
        if value >= 0.0:
            self.fitness = value

    @property
    def normalizedFitness(self):
        return self.normalizedFitness

    @normalizedFitness.setter
    def normalizedFitness(self, value):
        if 0.0 <= value <= 1.0:
            self.normalizedFitness = value

    def classification_did_finish(self):
        return self.isClassificationFinished

    def calc_fitness(self):
        pass

    def run(self):
        self.isClassificationFinished = False
        for gen in self.genes:
            if gen:
                # choose classifiers from list and execute
                pass
        self.calc_fitness()
        self.isClassificationFinished = True
