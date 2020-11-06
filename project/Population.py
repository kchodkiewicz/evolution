import random

from Phenotype import Phenotype


class Population:

    def __init__(self, size, committee, gen_length):
        self.size = size
        self.classifierCommittee = committee
        self.genLength = gen_length
        self.genNo = 0
        self.mutation_ratio = 3
        self.phenotypes = [Phenotype(self.classifierCommittee, gen_length) for i in range(self.size)]
        self.bestInGen = None

    def classification_did_finish(self):
        flag = True
        for i in self.phenotypes:
            if not i.is_classification_finished:
                flag = False
        return flag

    # Create list of indexes of true values in both parents
    # choose genes for child1 and remove them from true_genes
    # get list of genes that duplicate in both parents (AND)
    # add to child2 all duplicated genes and remaining genes from true_genes
    def cross(self, parent_first, parent_second):
        child_first = [False for _ in range(len(parent_first.genes))]
        child_second = [False for _ in range(len(parent_first.genes))]
        true_genes = []
        duplicates = []
        for i in range(len(parent_first.genes)):
            if parent_first.genes[i] or parent_second.genes[i]:
                true_genes.append(i)
            if parent_first.genes[i] and parent_second.genes[i]:
                duplicates.append(i)
        for _ in range(self.classifierCommittee):
            index = random.randint(0, len(true_genes) - 1)
            child_first[true_genes[index]] = True
            true_genes.pop(index)
        for index in true_genes:
            child_second[index] = True
        for index in duplicates:
            child_second[index] = True
        child1st = Phenotype(parent_first.committee, parent_first.gen_length)
        child2nd = Phenotype(parent_second.committee, parent_second.gen_length)
        child1st.genes = child_first
        child2nd.genes = child_second
        return child1st, child2nd

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
        mutate_ratio = random.randint(0, self.mutation_ratio)
        for _ in range(mutate_ratio):
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
        best = Phenotype(self.classifierCommittee, self.genLength)
        for phenotype in self.phenotypes:
            if phenotype.fitness > best.fitness:
                best = phenotype
        return best

    def find_parent(self):
        index = 0
        rand = random.uniform(0, 1)
        while rand > 0:
            rand = rand - self.phenotypes[index].normalizedFitness
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

        champ = self.find_best_in_gen()

        for phenotype in self.phenotypes:
            phenotype.normalizedFitness = phenotype.fitness / total_fitness

        new_generation = [Phenotype(self.classifierCommittee, self.genLength) for i in range(self.size)]

        # TODO multi-threading
        i = 0
        while (i + 1) < len(self.phenotypes):
            first_parent = self.find_parent()
            second_parent = self.find_parent()

            child1st, child2nd = self.cross(first_parent, second_parent)
            new_generation[i] = child1st
            new_generation[i + 1] = child2nd
            i += 2
        for phenotype in self.phenotypes:
            self.mutate(phenotype)

        new_generation[0] = champ

    def run(self):
        for phenotype in self.phenotypes:
            phenotype.run()
