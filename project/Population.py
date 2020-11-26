import random

from Phenotype import Phenotype


class Population(object):

    def __init__(self, size, committee, gen_length):
        self.size = size
        self.classifierCommittee = committee
        self.genLength = gen_length
        self.genNo = 0
        self.mutation_ratio = 0.3  # max amount of changed genes in phenotype
        self.phenotypes = [Phenotype(self.classifierCommittee, self.genLength) for i in range(self.size)]
        self.bestInGen = None
        self.genFitness = []

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
    def cross(self, parent_first, parent_second): # TODO fucked up shit
        child1st = Phenotype(self.classifierCommittee, self.genLength)
        child2nd = Phenotype(self.classifierCommittee, self.genLength)

        genes1st = []
        genes2nd = []

        for i, gen in enumerate(parent_first):
            if gen:
                genes1st.append(i)
        for i, gen in enumerate(parent_second):
            if gen:
                genes2nd.append(i)

        for gen in child1st.genes:


        return child1st, child2nd

        """
        child_first = [False for _ in range(len(parent_first.genes))]
        child_second = [False for _ in range(len(parent_first.genes))]
        true_genes = []
        duplicates = []

        for i in range(len(parent_first.genes)):
            if parent_first.genes[i] or parent_second.genes[i]:
                true_genes.append(i)
            if parent_first.genes[i] and parent_second.genes[i]:
                duplicates.append(i)
        print("true", true_genes, "len:", len(true_genes))
        print("dups", duplicates, "len:", len(duplicates))
        for _ in range(self.classifierCommittee):
            #  index = random.randrange(len(true_genes))
            if len(true_genes) > 0:
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
        child1st.genes = child_first
        child2nd.genes = child_second
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
                best = phenotype
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

        champ = self.find_best_in_gen()

        for phenotype in self.phenotypes:
            phenotype.normalizedFitness = phenotype.fitness / total_fitness

        new_generation = []

        # TODO multi-threading
        while len(new_generation) < len(self.phenotypes):
            first_parent = self.find_parent()
            second_parent = self.find_parent()

            i = 0
            j = 0
            for it in first_parent.genes:
                if it:
                    i += 1
            for jt in first_parent.genes:
                if jt:
                    j += 1
            print("first", i)
            print("second", j)

            child1st, child2nd = self.cross(first_parent, second_parent)

            new_generation.append(child1st)
            new_generation.append(child2nd)
        for phenotype in self.phenotypes:
            self.mutate(phenotype)
        new_generation.pop(0)
        new_generation.append(champ)
        self.phenotypes = new_generation
        print("Best in gen fitness:", self.bestInGen.fitness)

    # Check whether all phenotypes are getting similar fitness
    # i.e. max fitness is close to avg fitness
    # If so then increase mutation ratio to eliminate similarity
    def validate(self):
        sum_fitness = 0
        for phenotype in self.phenotypes:
            sum_fitness += phenotype.fitness
        avg_fitness = sum_fitness / len(self.phenotypes)
        if (self.bestInGen.fitness * 0.9) < avg_fitness:
            self.mutation_ratio = 0.7
        else:
            self.mutation_ratio = 0.3

    def run(self):
        print("Gen No", self.genNo)
        for phenotype in self.phenotypes:
            fit = phenotype.run()
            self.genFitness.append(fit)
        self.genNo += 1
        print("Gen fitness", self.genFitness)

    def test(self):
        self.phenotypes[0].test()
