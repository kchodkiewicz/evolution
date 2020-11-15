from Population import Population
from models.Model import Model


def fitness_is_progressing():
    score_sum = sum(fitness_scores[len(fitness_scores) - 20:])
    score_avg = score_sum / 20
    if score_avg == sorted(fitness_scores, reverse=True)[0]:
        return False
    return True


if __name__ == '__main__':
    # Values passed by user
    dataset = "dataset.csv"
    col_name = "column"
    metrics = "f1_score"  # accuracy_score, auc, f1_score. Default f1

    print("Hello")

    model = Model(dataset, col_name, metrics)
    # TODO add loadPopulation method
    population = Population(size=10, committee=10, gen_length=100)

    fitness_scores = []

    while fitness_is_progressing():
        if population.classification_did_finish():
            population.select()
            population.validate()
        else:
            population.run()
            fitness_scores.append(population.bestInGen.fitness)

