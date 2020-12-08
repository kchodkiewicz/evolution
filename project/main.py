import json
import time

from Population import Population, conv_genes
from models.Model import Model
import pandas as pd
from sklearn.model_selection import train_test_split

from models.instances import Instances


def fitness_is_progressing():
    if len(fitness_scores) >= 20:
        score_sum = sum(fitness_scores[len(fitness_scores) - 20:])
        score_avg = score_sum / 20
        score_max = sorted(fitness_scores, reverse=True)[0]
        if abs(score_avg - score_max) < 0.01:
            return False
    return True


if __name__ == '__main__':
    # Values passed by user
    dataset = "csv_result-PhishingData.csv"
    col = "Result"
    metrics = "accuracy_score"  # accuracy_score, auc, f1_score. Default f1

    Model.dataset = pd.read_csv(dataset)
    #  Model.METRICS_METHOD = metrics

    X = Model.dataset.drop(columns=col)
    X = X.drop(columns="id")
    y = Model.dataset[col]
    Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    inst = Instances()
    inst.trainClassifiers(Model.X_train, Model.y_train)

    population = Population(size=100, committee=10, gen_length=len(inst.trained_classifiers))
    # loading genes from file
    #  population.load_population("output_files/population_dump/2020-12-8_19:29:51.json")
    fitness_scores = []

    while True:
        population.run()
        population.select()
        population.validate()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing():
            print(conv_genes(population.bestInGen.genes))
            with open(f"output_files/classifiers_scores/{time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]}_"
                      f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}.json", "w") as f:
                json.dump(population.genFitness, f, indent=4)
            break

