import copy
import json
import math
import random
import time
import multiprocessing as mp
from Phenotype import Phenotype
from utils import print_progress, conv_genes, write_to_json
from models.Model import Model


class Population(object):

    def __init__(self, size, committee, gen_length):
        self.size = size
        self.classifierCommittee = committee
        self.genLength = gen_length
        self.genNo = 0
        if gen_length < committee:
            raise ValueError
        self.mutation_ratio = 1/gen_length  # max amount of changed genes in phenotype
        self.phenotypes = [Phenotype(i, self.classifierCommittee, self.genLength) for i in range(self.size)]
        self.bestInGen = None
        self.__genFitness = []
        self.output = {}
        self.start_time = time.localtime()
        self.__validation_res = {}

    def load_population(self, filename):
        try:
            with open(filename) as f:
                file_genes = json.load(f)
        except FileNotFoundError as e:
            print(e)
        else:
            for i, phenotype in enumerate(self.phenotypes):
                phenotype.genes = file_genes[str(i)]

    @property
    def genFitness(self):
        return self.__genFitness

    # Create list of indexes of true values in both parents
    # choose genes for child1 and remove them from true_genes
    # get list of genes that duplicate in both parents (AND)
    # add to child2 all duplicated genes and remaining genes from true_genes
    # TODO crossing - test it (cut points) !!!
    def cross(self, cross_id, parent_first, parent_second):
        child1st = Phenotype(cross_id, self.classifierCommittee, self.genLength)
        child2nd = Phenotype(cross_id + 1, self.classifierCommittee, self.genLength)
        cut_point1 = random.randint(1, self.genLength - 2)
        cut_point2 = random.randint(1, self.genLength - 2)
        if cut_point1 == cut_point2:
            cut_point2 = cut_point2 + random.randint(1, self.genLength - 1 - cut_point2)
        genes1 = []
        genes2 = []
        genes1.append(parent_first.genes[0:cut_point1].copy())
        genes1.append(parent_second.genes[cut_point1 + 1:cut_point2].copy())
        genes1.append(parent_first.genes[cut_point2 + 1:].copy())
        genes2.append(parent_second.genes[0:cut_point1].copy())
        genes2.append(parent_first.genes[cut_point1 + 1:cut_point2].copy())
        genes2.append(parent_second.genes[cut_point2 + 1:].copy())

        child1st.genes = genes1
        child2nd.genes = genes2
        it = 0
        for i in child1st.genes:
            if i:
                it += 1
        child1st.committee = it
        jt = 0
        for j in child1st.genes:
            if j:
                jt += 1
        child2nd.committee = jt

        return child1st, child2nd

    # Create lists of True and False values in genes
    # Get random amount of mutations (between 0 and 3)
    # Get random indexes of values in lists of True and False values
    # Remove value from list of True values and add it to the list of False values
    # Then repeat for list of False values
    # Create list of genes according to the modified positive_values and negative_values
    def mutate(self, phenotype):
        mutate_ratio = random.uniform(0, self.mutation_ratio)
        for _ in range(int(mutate_ratio * phenotype.committee)):
            index = random.randint(0, len(phenotype.genes) - 1)
            phenotype.genes[index] = not phenotype.genes[index]
        it = 0
        for i in phenotype.genes:
            if i:
                it += 1
        phenotype.committee = it

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
    def find_parent(self, sorted_phenotypes):
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
        sorted_phenotypes = sorted(self.phenotypes,
                                   key=lambda p: phenotype.normalizedFitness, reverse=True)
        new_generation = []
        i = 0
        while len(new_generation) < len(self.phenotypes):
            print_progress(i + 2, len(self.phenotypes), "Crossing")
            first_parent = copy.deepcopy(self.find_parent(sorted_phenotypes))
            second_parent = copy.deepcopy(self.find_parent(sorted_phenotypes))
            child1st, child2nd = self.cross(i, first_parent, second_parent)
            new_generation.append(copy.deepcopy(child1st))
            new_generation.append(copy.deepcopy(child2nd))
            i += 2
        print('')
        for j, phenotype in enumerate(self.phenotypes):
            print_progress(j + 1, len(self.phenotypes), "Mutating")
            self.mutate(phenotype)
        print('')
        new_generation.pop(0)
        new_generation.append(copy.deepcopy(self.bestInGen))
        self.phenotypes = copy.deepcopy(new_generation)
        if Model.verbose:
            print("Best in gen: fitness =", self.bestInGen.fitness, "genes =", conv_genes(self.bestInGen.genes))
        self.output[self.genNo] = conv_genes(self.bestInGen.genes)
        if self.genNo % 5 == 0:
            write_to_json("gen_stats", self.output)
            specimen = {}
            for i, phenotype in enumerate(self.phenotypes):
                specimen[i] = phenotype.genes
            write_to_json("population_dump", specimen)

    # Check whether all phenotypes are getting similar fitness
    # i.e. max fitness is close to avg fitness
    # If so then increase mutation ratio to eliminate similarity
    def validate(self):
        sum_fitness = 0
        self.find_best_in_gen()
        for phenotype in self.phenotypes:
            sum_fitness += phenotype.fitness
        avg_fitness = sum_fitness / len(self.phenotypes)
        self.__validation_res[self.genNo] = {}
        self.__validation_res[self.genNo]["avg"] = avg_fitness
        self.__validation_res[self.genNo]["max"] = self.bestInGen.fitness
        write_to_json("validation_res", self.__validation_res)
        if (self.bestInGen.fitness * 0.9) < avg_fitness:
            self.mutation_ratio = 10/self.genLength
        else:
            self.mutation_ratio = 1/self.genLength

    def run_normally(self):
        print("Gen No", self.genNo, end=' ', flush=True)
        for phenotype in self.phenotypes:
            fit = phenotype.run()
            self.__genFitness.append(fit)
        self.genNo += 1
