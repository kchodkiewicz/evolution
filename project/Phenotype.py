import random


class Phenotype:

    def __init__(self, committee, gen_length):
        self.committee = committee
        self.genLength = gen_length
        self.genes = [False for x in range(self.genLength)]
        self.fitness = 0.0
        self.isBest = False
        self.isClassificationFinished = False
        self.create_random_genes()

    # generate 10 positive genes (classifiers)
    def create_random_genes(self):
        it = 0
        while it < self.committee:
            index = random.randint(0, self.genLength - 1)
            if not self.genes[index]:
                self.genes[index] = True
                it += 1

    def load_genes(self, genes):
        self.genes = genes

    def calc_fitness(self):
        pass

    def run(self):
        for gen in self.genes:
            if gen:
                # choose classifiers from list and execute
                pass
