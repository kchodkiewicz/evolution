import random

from Phenotype import Phenotype


class Population:

    def __init__(self, size, committee, gen_length):
        self.size = size
        self.classifierCommittee = committee
        self.mutation_ratio = 3
        self.phenotypes = [Phenotype(self.classifierCommittee, gen_length) for i in range(self.size)]
        self.bestInGen = None

    def classification_did_finish(self):
        flag = True
        for i in self.phenotypes:
            if not i.isClassificationFinished:
                flag = False
        return flag

    # Create list of indexes of true values in both parents
    # choose genes for child1 and remove them from true_genes
    # get list of genes that duplicate in both parents (AND)
    # add to child2 all duplicated genes and remaining genes from true_genes
    def cross(self, father, mother):
        child1 = [False for _ in range(len(father.genes))]
        child2 = [False for _ in range(len(father.genes))]
        true_genes = []
        duplicates = []
        for i in range(0, len(father.genes)):
            if father.genes[i] or mother.genes[i]:
                true_genes.append(i)
            if father.genes[i] and mother.genes[i]:
                duplicates.append(i)
        for _ in range(self.classifierCommittee):
            index = random.randint(0, len(true_genes) - 1)
            child1[true_genes[index]] = True
            true_genes.pop(index)
        for index in true_genes:
            child2[index] = True
        for index in duplicates:
            child2[index] = True
        father.load_genes(child1)
        mother.load_genes(child2)

    # Create lists of True and False values in genes
    # Get random amount of mutations (between 0 and 3)
    # Get random indexes of values in lists of True and False values
    # Remove value from list of True values and add it to the list of False values
    # Then repeat for list of False values
    # Create list of genes according to the modified positive_values and negative_values
    def mutate(self, phenotype):
        positive_values = []
        negative_values = []
        for i in range(0, len(phenotype.genes) - 1):
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

    # Create sorted (based on fitness) list of all phenotypes
    # Generate weighted list with best phenotypes having most slots
    # Cross with randomly chosen parents from weighted list
    # Mutate phenotypes
    def select(self):
        pass

    def run(self):
        pass
