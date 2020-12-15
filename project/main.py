import json
import time
import keyboard

from Population import Population, conv_genes, write_to_json
from models.Model import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from models.instances import Instances, predictSelected


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
    Model.X_train, X_in, Model.y_train, y_in = train_test_split(X, y, test_size=0.3)
    Model.X_test, Model.X_validate, Model.y_test, Model.y_validate = train_test_split(X_in, y_in, test_size=1/3)
    """
    dataset = "dataset-har-PUC-Rio-ugulino.csv"
    col = "class"
    metrics = "accuracy_score"  # accuracy_score, auc, f1_score. Default f1
    Model.dataset = pd.read_csv(dataset, sep=';', decimal=',')
    #  Model.METRICS_METHOD = metrics
    X = Model.dataset.drop(columns=col)
    X['gender'] = X['gender'].map({'Woman': 1, 'Man': 0})
    X = X.drop(columns="user")
    y = Model.dataset[col]
    Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    """

    inst = Instances()
    inst.trainClassifiers(Model.X_train, Model.y_train)
    inst.predictClassifiers(Model.X_test)

    """
    population = Population(size=100, committee=10, gen_length=len(inst.trained_classifiers))
    # loading genes from file
    #  population.load_population("output_files/population_dump/2020-12-8_19:29:51.json")
    fitness_scores = []
    
    while True:
        population.run_normally(False)
        # population.run_async(4)
        population.validate()
        population.select()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing():
            break
    print(conv_genes(population.bestInGen.genes))
    write_to_json("classifiers_scores", population.genFitness)
    """

    population = Population(size=100, committee=10, gen_length=len(inst.trained_classifiers))
    fitness_scores = []
    while True:
        population.run_normally()
        # population.run_async(4)
        population.validate()
        population.select()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing():
            break

    final_models = []
    for i, gen in enumerate(population.bestInGen.genes):
        if gen:
            final_models.append(inst.trained_classifiers[i])
    final_predicts = predictSelected(final_models, Model.X_validate)
    committee_answers = []
    for i in range(len(final_predicts)):
        tmp = {}
        for predicts in final_predicts:
            if predicts[i] in tmp.keys():
                tmp[predicts[i]] += 1
            else:
                tmp[predicts[i]] = 1
        inverse = [(value, key) for key, value in tmp.items()]
        val = max(inverse)[1]
        committee_answers.append(val)
    model = Model()
    score = model.calcScore(predictions=committee_answers)


    print("Classifiers:", conv_genes(population.bestInGen.genes))
    print("Score:", score)
    print("Separate scores:", inst.scores)
    write_to_json("classifiers_scores", population.genFitness)


