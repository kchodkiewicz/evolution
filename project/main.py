from Population import Population
from models.Model import Model

import pandas as pd
from sklearn.model_selection import train_test_split


def fitness_is_progressing():
    score_sum = sum(fitness_scores[len(fitness_scores) - 20:])
    score_avg = score_sum / 20
    if score_avg == sorted(fitness_scores, reverse=True)[0]:
        return False
    return True


if __name__ == '__main__':
    # Values passed by user
    dataset = "iris.csv"
    col = "Species"
    metrics = "f1_score"  # accuracy_score, auc, f1_score. Default f1

    print("Hello")

    #model = Model()
    Model.dataset = pd.read_csv(dataset)
    Model.METRICS_METHOD = metrics

    X = Model.dataset.drop(columns=col)
    y = Model.dataset[col]
    Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(X, y, test_size=0.2)

    # TODO add loadPopulation method
    population = Population(size=1000, committee=10, gen_length=60)

    fitness_scores = []

    while True:
        if population.classification_did_finish():
            population.select()
            population.validate()
        else:
            population.run()
            fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing():
            break

