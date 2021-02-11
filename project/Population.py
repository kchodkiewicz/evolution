# Phenotypes list and methods for evolution -- its me and its good
import copy
import json
import random
import time
from Phenotype import Phenotype
from models.instances import Instances
from utils import print_progress, conv_genes, write_to_json
from models.Model import Model


class Population(object):

    def __init__(self, size, committee, gen_length):
        """
        Create population of phenotypes for genetic evolution

        :param size: number of phenotypes in population
        :param committee: amount of classifiers in committee
        :param gen_length: length of genotype
        """
        self.size = size
        self.classifierCommittee = committee
        self.genLength = gen_length
        self.genNo = 0
        if gen_length < committee:
            raise ValueError
        self.mutation_ratio = 1  # amount of swapped genes in phenotype
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
            print('\033[93m' + str(e) + '\033[0m')
        else:
            for i, phenotype in enumerate(self.phenotypes):
                phenotype.genes = file_genes[str(i)]

    @property
    def genFitness(self):
        return self.__genFitness

    def cross(self, cross_id, parent_first, parent_second):
        """
        Cross two phenotypes and create two new phenotypes

        Return two new phenotypes with mixed parents' genes
        :param cross_id: identification number of crossing
        :param parent_first: 1st parent phenotype
        :param parent_second: 2nd parent phenotype
        :rtype: (Phenotype, Phenotype, int, int)
        :return: (1st offspring, 2nd offspring, 1st cut point, 2nd cut point)
        """
        child1st = Phenotype(cross_id, self.classifierCommittee, self.genLength)
        child2nd = Phenotype(cross_id + 1, self.classifierCommittee, self.genLength)
        cut_point1 = random.randint(1, self.genLength - 3)
        cut_point2 = random.randint(cut_point1 + 1, self.genLength - 2)
        genes1 = []
        genes2 = []
        genes1[0:cut_point1] = parent_first.genes[0:cut_point1].copy()
        genes1[cut_point1:cut_point2] = parent_second.genes[cut_point1:cut_point2].copy()
        genes1[cut_point2:] = parent_first.genes[cut_point2:].copy()
        genes2[0:cut_point1] = parent_second.genes[0:cut_point1].copy()
        genes2[cut_point1:cut_point2] = parent_first.genes[cut_point1:cut_point2].copy()
        genes2[cut_point2:] = parent_second.genes[cut_point2:].copy()

        child1st.genes = genes1
        child2nd.genes = genes2
        it = 0
        for i in child1st.genes:
            if i:
                it += 1
        child1st.committee = it
        jt = 0
        for j in child2nd.genes:
            if j:
                jt += 1
        child2nd.committee = jt
        return child1st, child2nd, cut_point1, cut_point2

    def mutate(self, phenotype):
        """
        Mutate genotype of provided phenotype

        Invert one True and one False gene
        i.e. change one classifier in committee
        :param phenotype: currently mutating phenotype
        :return: None
        """
        for i in range(self.mutation_ratio):
            index1 = random.randint(0, len(phenotype.genes) - 1)
            index2 = random.randint(0, len(phenotype.genes) - 1)
            while index1 == index2:
                index2 = random.randint(0, len(phenotype.genes) - 1)
            while phenotype.genes[index1] == phenotype.genes[index2]:
                index2 = random.randint(0, len(phenotype.genes) - 1)
            tmp = phenotype.genes[index2]
            phenotype.genes[index2] = phenotype.genes[index1]
            phenotype.genes[index1] = tmp
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

    def roulette_wheel_selection(self):
        """
        Roulette wheel selection

        Select random number between [0, 1],
        then return phenotype if its fitness is higher
        or go to next phenotype
        :rtype: Phenotype
        :return: parent selected for crossing
        """
        rand = random.uniform(0, 1)
        for phenotype in self.phenotypes:
            if phenotype.normalizedFitness > rand:
                phenotype.counter = phenotype.counter + 1
                return phenotype
        self.phenotypes[0].counter = self.phenotypes[0].counter + 1
        return self.phenotypes[0]

    def tournament_selection(self):
        """
        Tournament selection

        Randomly select two phenotypes,
        then return one with higher fitness
        :rtype: Phenotype
        :return: parent selected for crossing
        """
        candidate1 = self.phenotypes[random.randint(0, len(self.phenotypes) - 1)]
        candidate2 = self.phenotypes[random.randint(0, len(self.phenotypes) - 1)]

        while candidate1 == candidate2:
            candidate2 = self.phenotypes[random.randint(0, len(self.phenotypes) - 1)]

        if candidate1.fitness > candidate2.fitness:
            candidate1.counter = candidate1.counter + 1
            return candidate1
        else:
            candidate2.counter = candidate2.counter + 1
            return candidate2

    def select(self):
        """
        Create sorted (based on fitness) list of all phenotypes
        Generate weighted list with best phenotypes having most slots
        Cross with randomly chosen parents from weighted list
        Mutate phenotypes
        :return: None
        """
        total_fitness = 0.0
        for phenotype in self.phenotypes:
            total_fitness += phenotype.fitness

        for phenotype in self.phenotypes:
            phenotype.normalizedFitness = phenotype.fitness / total_fitness

        new_generation = []
        i = 0
        while len(new_generation) < len(self.phenotypes):
            print_progress(i + 2, len(self.phenotypes), "Crossing")
            first_parent = copy.deepcopy(self.tournament_selection())
            second_parent = copy.deepcopy(self.tournament_selection())
            while first_parent == second_parent:
                first_parent = copy.deepcopy(self.tournament_selection())
            child1st, child2nd, c1, c2 = self.cross(i, first_parent, second_parent)
            new_generation.append(copy.deepcopy(child1st))
            new_generation.append(copy.deepcopy(child2nd))
            i += 2
        if Model.VERBOSE:
            print('')
        for j, phenotype in enumerate(self.phenotypes):
            print_progress(j + 1, len(self.phenotypes), "Mutating")
            self.mutate(phenotype)
        if Model.VERBOSE:
            print('')
        new_generation.pop(0)
        new_generation.append(copy.deepcopy(self.bestInGen))
        self.phenotypes = copy.deepcopy(new_generation)
        if Model.VERBOSE:
            print(f"Best in gen {self.genNo}: fitness =", self.bestInGen.fitness, "genes =",
                  conv_genes(self.bestInGen.genes))
        self.output[self.genNo] = conv_genes(self.bestInGen.genes)
        write_to_json("gen_stats", self.output)
        if self.genNo % 5 == 0:
            pred_copy = Instances.predictions_arr.copy()
            specimen = {'used_models': Instances.models_index,
                        'predictions': [0 for _ in range(len(pred_copy))]}
            for tt, dim1 in enumerate(pred_copy):
                specimen['predictions'][tt] = dim1.tolist()

            for i, phenotype in enumerate(self.phenotypes):
                specimen[i] = phenotype.genes
            write_to_json("population_dump", specimen)

    def validate(self):
        """
        Check whether all phenotypes are getting similar fitness
        i.e. max fitness is close to avg fitness
        If so then increase mutation ratio to eliminate similarity
        :return: None
        """
        sum_fitness = 0
        self.find_best_in_gen()
        for phenotype in self.phenotypes:
            sum_fitness += phenotype.fitness
        avg_fitness = sum_fitness / len(self.phenotypes)
        self.__validation_res[self.genNo] = {}
        self.__validation_res[self.genNo]["avg"] = avg_fitness
        self.__validation_res[self.genNo]["max"] = self.bestInGen.fitness
        write_to_json("validation_res", self.__validation_res)
        if self.genNo < 15:
            if (self.bestInGen.fitness * 0.999) < avg_fitness:
                self.mutation_ratio = 2
            else:
                self.mutation_ratio = 1

    def run_normally(self):
        if Model.VERBOSE:
            print("", end=' ', flush=True)
        for i, phenotype in enumerate(self.phenotypes):
            print_progress(i + 1, len(self.phenotypes), "Selecting")
            fit = phenotype.run()
            self.__genFitness.append(fit)
        if Model.VERBOSE:
            print('')
        self.genNo += 1
