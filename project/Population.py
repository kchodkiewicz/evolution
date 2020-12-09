import copy
import json
import random
import sys
import time
from time import sleep

from matplotlib.pyplot import plot
from Phenotype import Phenotype


# Return indexes of true genes
def conv_genes(genes_bool):
    arr = []
    for i, gen in enumerate(genes_bool):
        if gen:
            arr.append(i)
    return arr


class Population(object):

    def __init__(self, size, committee, gen_length):
        self.size = size
        self.classifierCommittee = committee
        self.genLength = gen_length
        self.genNo = 0
        self.mutation_ratio = 0.3  # max amount of changed genes in phenotype
        self.phenotypes = [Phenotype(self.classifierCommittee, self.genLength) for i in range(self.size)]
        self.bestInGen = None
        self.__genFitness = []
        self.output = {}
        self.start_time = time.localtime()

    def load_population(self, filename):
        with open(filename) as f:
            file_genes = json.load(f)
        for i, phenotype in enumerate(self.phenotypes):
            phenotype.genes = file_genes[str(i)]

    @property
    def genFitness(self):
        return self.__genFitness

    def classification_did_finish(self):
        for i in self.phenotypes:
            if not i.is_classification_finished:
                break
        else:
            return True
        return False

    # Create list of indexes of true values in both parents
    # choose genes for child1 and remove them from true_genes
    # get list of genes that duplicate in both parents (AND)
    # add to child2 all duplicated genes and remaining genes from true_genes
    def cross(self, parent_first, parent_second):  # TODO tell me why

        child1st = Phenotype(self.classifierCommittee, self.genLength)
        child2nd = Phenotype(self.classifierCommittee, self.genLength)
        genes1 = []
        genes2 = []
        genes1.append(parent_first.genes[0:20].copy())
        genes1.append(parent_second.genes[21:40].copy())
        genes1.append(parent_first.genes[41:].copy())
        genes2.append(parent_second.genes[0:20].copy())
        genes2.append(parent_first.genes[21:40].copy())
        genes2.append(parent_second.genes[41:].copy())

        child1st.genes = genes1
        child2nd.genes = genes2
        it = 0
        for i in child1st.genes:
            if i:
                it += 1
        jt = 0
        for j in child1st.genes:
            if j:
                jt += 1

        return child1st, child2nd
        """
        child_first = [False for _ in range(len(parent_first.genes))]
        child_second = [False for _ in range(len(parent_first.genes))]
        true_genes = []
        duplicates = []

        for i in range(len(parent_first.genes)):
            if parent_first.genes[i] or parent_second.genes[i]:
                true_genes.append(i)
        for i in range(len(parent_first.genes)):
            if parent_first.genes[i] and parent_second.genes[i]:
                duplicates.append(i)
        print("true", true_genes, "len:", len(true_genes))
        print("dups", duplicates, "len:", len(duplicates))
        for _ in range(self.classifierCommittee):
            #  index = random.randrange(len(true_genes))
            #if len(true_genes) > 0:
            random.shuffle(true_genes)
            index = true_genes.pop()
            child_first[index] = True
            #  true_genes.pop(index)
        for index in true_genes:
            child_second[index] = True
        for index in duplicates:
            child_second[index] = True
        child1st = Phenotype(committee=self.classifierCommittee, gen_length=self.genLength)
        child2nd = Phenotype(committee=self.classifierCommittee, gen_length=self.genLength)
        child1st.genes = child_first.copy()
        child2nd.genes = child_second.copy()
        return child1st, child2nd
        """

    # Create lists of True and False values in genes
    # Get random amount of mutations (between 0 and 3)
    # Get random indexes of values in lists of True and False values
    # Remove value from list of True values and add it to the list of False values
    # Then repeat for list of False values
    # Create list of genes according to the modified positive_values and negative_values
    def mutate(self, phenotype):
        positive_values = []
        negative_values = []
        for i in range(len(phenotype.genes)):
            if phenotype.genes[i]:
                positive_values.append(i)
            else:
                negative_values.append(i)
        mutate_ratio = random.uniform(0, self.mutation_ratio)
        for _ in range(int(mutate_ratio * self.classifierCommittee)):
            positive_index = random.randint(0, len(positive_values) - 1)
            negative_index = random.randint(0, len(negative_values) - 1)
            negative_values.append(positive_values[positive_index])
            positive_values.pop(positive_index)
            positive_values.append(negative_values[negative_index])
            negative_values.pop(negative_index)
        for index in positive_values:
            phenotype.genes[index] = True
        for index in negative_values:
            phenotype.genes[index] = False

    def find_best_in_gen(self):
        best = self.phenotypes[0]
        for phenotype in self.phenotypes:
            if phenotype.fitness > best.fitness:
                best = copy.deepcopy(phenotype)
        self.bestInGen = best
        return best

    # Pick random phenotype from a weighted list of phenotypes
    # Get random number between 0 and 1
    # Check if phenotype's normalized fitness is greater than random number
    # If it is then return this phenotype
    # If not then try for another phenotype (keep the same random number)
    def find_parent(self):
        sorted_phenotypes = sorted(self.phenotypes,
                                   key=lambda phenotype: phenotype.normalizedFitness, reverse=True)
        index = 0
        rand = random.uniform(0, 1)
        while rand > 0:
            if index >= len(sorted_phenotypes):
                return self.phenotypes[0]
            rand = rand - sorted_phenotypes[index].normalizedFitness
            index += 1

        return self.phenotypes[index - 1]

    # Create sorted (based on fitness) list of all phenotypes
    # Generate weighted list with best phenotypes having most slots
    # Cross with randomly chosen parents from weighted list
    # Mutate phenotypes
    def select(self):
        total_fitness = 0.0
        for phenotype in self.phenotypes:
            total_fitness += phenotype.fitness

        for phenotype in self.phenotypes:
            phenotype.normalizedFitness = phenotype.fitness / total_fitness

        new_generation = []
        i = 0
        while len(new_generation) < len(self.phenotypes):
            first_parent = copy.deepcopy(self.find_parent())
            second_parent = copy.deepcopy(self.find_parent())
            child1st, child2nd = self.cross(first_parent, second_parent)
            new_generation.append(copy.deepcopy(child1st))
            new_generation.append(copy.deepcopy(child2nd))
            sys.stdout.write("Crossing: [{0} / {1}]   \r".format(i + 2, len(self.phenotypes)))
            sys.stdout.flush()
            sleep(0.05)
            i += 2
        for j, phenotype in enumerate(self.phenotypes):
            self.mutate(phenotype)
            sys.stdout.write("Mutating: [{0} / {1}]   \r".format(j + 1, len(self.phenotypes)))
            sys.stdout.flush()
            sleep(0.05)
        new_generation.pop(0)
        new_generation.append(copy.deepcopy(self.bestInGen))
        self.phenotypes = copy.deepcopy(new_generation)
        print("Best in gen fitness:", self.bestInGen.fitness)
        self.output[self.genNo] = conv_genes(self.bestInGen.genes)
        if self.genNo % 5 == 0:
            with open(f"output_files/gen_stats/{self.start_time[0]}-{self.start_time[1]}-{self.start_time[2]}_"
                      f"{self.start_time[3]}:{self.start_time[4]}:{self.start_time[5]}.json", "w") as gen_stats:
                json.dump(self.output, gen_stats, indent=4)
            specimen = {}
            for i, phenotype in enumerate(self.phenotypes):
                specimen[i] = phenotype.genes
            with open(f"output_files/population_dump/{self.start_time[0]}-{self.start_time[1]}-{self.start_time[2]}_"
                      f"{self.start_time[3]}:{self.start_time[4]}:{self.start_time[5]}.json", "w") as gen_stats:
                json.dump(specimen, gen_stats, indent=4)

    # Check whether all phenotypes are getting similar fitness
    # i.e. max fitness is close to avg fitness
    # If so then increase mutation ratio to eliminate similarity
    def validate(self):
        sum_fitness = 0
        champ = self.find_best_in_gen()
        for phenotype in self.phenotypes:
            sum_fitness += phenotype.fitness
        avg_fitness = sum_fitness / len(self.phenotypes)
        print("avg", avg_fitness, "max", self.bestInGen.fitness)
        if (self.bestInGen.fitness * 0.9) < avg_fitness:
            self.mutation_ratio = 0.7
        else:
            self.mutation_ratio = 0.3

    def run(self):
        print("Gen No", self.genNo)
        for phenotype in self.phenotypes:
            fit = phenotype.run()
            self.__genFitness.append(fit)
        self.genNo += 1

    def test(self):
        self.phenotypes[0].test()
