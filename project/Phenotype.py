import random


class Phenotype:

    committee = 0
    genes = [False for x in range(100)]
    fitness = 0.0
    isBest = False
    isClassificationFinished = False

    def __init__(self, committee):
        self.committee = committee
        self.create_random_genes()

    # generate 10 positive genes (classifiers)
    def create_random_genes(self):
        it = 0
        while it < self.committee:
            index = random.randint(0, 99)
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
