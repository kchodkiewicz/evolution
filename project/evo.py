# Main -- its me and its good
import json
import os
import pickle
import sys
import time
from random import randint
import pandas as pd
import numpy as np
from pandas import errors
from sklearn.model_selection import train_test_split
from Population import Population, conv_genes, write_to_json
from models.Model import Model
from models.instances import Instances, trainClassifiers, predictClassifiers
from utils import parse_args, variance_threshold_selector, fitness_is_progressing, predictSelected, vote, clear_cache, \
    print_progress
from plotting import plot_scores_progress, plot_best_phenotype_genes_progress, plot_genes_in_last_gen, \
    plot_avg_max_distance_progress

if __name__ == '__main__':
    dataset, col, metrics, pop, comm, load_file, verbose, testing, pre_trained = parse_args(sys.argv[1:])

    inst = Instances()
    model = Model()

    # Add id for run
    timestamp = time.time()
    Model.RUN_ID = str(timestamp).replace('.', '-')
    print('Run identifier:', Model.RUN_ID)
    Model.verbose = verbose
    Model.TEST = testing
    Model.PRE_TRAIN = pre_trained
    Model.METRICS_METHOD = metrics
    clear_cache()

    try:
        Model.dataset = pd.read_csv(dataset)
    except pd.errors.ParserError:
        print('\033[93m' + "Incorrect path to file" + '\033[0m')
        sys.exit(2)

    # remove features with low variance (same value in 90% of samples)
    variance_threshold_selector(model.dataset, 0.9)
    # split for X, y and remove id column
    try:
        X = model.dataset.drop(columns=col)
    except AttributeError:
        print('\033[93m' + "Column with name: " + str(col) + " not found in provided dataset" + '\033[0m')
        sys.exit(2)
    except KeyError:
        print('\033[93m' + "Column with name: " + str(col) + " not found in provided dataset" + '\033[0m')
        sys.exit(2)

    X = X.drop(columns="id", errors='ignore')
    X = X.drop(columns="Id", errors='ignore')
    X = X.drop(columns="ID", errors='ignore')
    X = X.drop(columns="Identification", errors='ignore')
    X = X.drop(columns="identification", errors='ignore')
    y = model.dataset[col]

    Model.X_train, X_in, Model.y_train, y_in = train_test_split(X, y, test_size=0.3)
    Model.X_test, Model.X_validate, Model.y_test, Model.y_validate = train_test_split(X_in, y_in, test_size=0.3)

    print("Running for configuration:", "\n* dataset:", dataset, "\n* column:", col, "\n* metrics method:", metrics,
          "\n* population size:", pop, "\n* committee size:", comm, "\n* load_genes:", load_file)

    if load_file is None:
        # if running normal mode
        trainClassifiers(model.X_train, model.y_train)
        predictClassifiers(model.X_test)
    else:
        # if continuing previous evolution use models_list and predictions from .json file
        try:
            with open(load_file) as f:
                file_genes = json.load(f)
        except FileNotFoundError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode. Error: " + str(e) + '\033[0m')
        except ValueError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode. Error: " + str(e) + '\033[0m')
        except KeyError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode. Error: " + str(e) + '\033[0m')
        else:
            Instances.models_index = file_genes['used_models']
            ndarray_compatible = np.asarray(file_genes['predictions'])
            Instances.predictions_arr = ndarray_compatible
    try:
        if comm > len(Instances.predictions_arr):
            raise ValueError
        population = Population(size=pop, committee=comm, gen_length=len(Instances.predictions_arr))
    except ValueError:
        print('\033[93m' + "Committee size cannot be greater than the amount of different classifiers (" +
              str(len(Instances.predictions_arr)) + ")" + '\033[0m')
        sys.exit(2)

    if load_file is not None:
        try:
            population.load_population(load_file)
        except FileNotFoundError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')
        except ValueError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')
        except KeyError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')

    # EVOLUTION --------------------------------------------------------
    fitness_scores = []
    while True:
        population.run_normally()
        population.validate()
        population.select()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing(fitness_scores):
            break
    print('\033[94m' + 'Evolution finished' + '\033[0m')

    # CALCULATE METRICS, MAKE REPORTS ----------------------------------
    # Get specified untrained classifier from file
    def load_vanilla_classifier(it):
        try:
            with open(os.path.join('models/vanilla_classifiers', f'v-{inst.models_index[it]}.pkl'), 'rb') as fid:
                instance = pickle.load(fid)
        except FileNotFoundError as ex:
            print('\033[93m' + str(ex) + '\033[0m')
            print('\033[93m' + f"Cannot include model at index {it} in final score" + '\033[0m')
        else:
            return instance


    # SCORE OF EVOLVED MODELS ------------------------------------------
    # create final list of models
    final_models = []
    for i, gen in enumerate(population.bestInGen.genes):
        if gen:
            print_progress(i + 1, len(population.bestInGen.genes), 'Calculating final score: ')
            fitted = load_vanilla_classifier(i).fit(model.X_train, model.y_train)
            final_models.append(fitted)
    print('')
    score, report = vote(final_models, model.X_validate)

    # SEPARATE SCORES OF EVOLVED MODELS --------------------------------
    # create list of models used in final list
    separated_models = []
    for i, mod in enumerate(conv_genes(population.bestInGen.genes)):
        print_progress(i + 1, len(conv_genes(population.bestInGen.genes)), 'Calculating separate scores: ')
        fitted = load_vanilla_classifier(i).fit(model.X_train, model.y_train)
        separated_models.append(fitted)
    print('')
    # make predictions for every model
    separated_predicts = predictSelected(separated_models, model.X_validate)
    # calculate separate score for every model
    separated_scores = []
    for mod in separated_predicts:
        separated_scores.append(model.calcScore(mod, verify=True))

    # SCORE OF FIRST 10 / RANDOM MODELS --------------------------------
    # create theoretical list of models
    theoretical_models = []
    if Model.TEST:
        for i in range(comm):
            print_progress(i + 1, comm, 'Calculating theoretical score: ')
            fitted = load_vanilla_classifier(i).fit(model.X_train, model.y_train)
            theoretical_models.append(fitted)
        theoretical_score, theoretical_report = vote(theoretical_models, model.X_validate)
    else:
        for i in range(comm):
            print_progress(i, comm, 'Calculating theoretical score: ')
            fitted = load_vanilla_classifier(randint(0, len(Instances.predictions_arr) - 1)).fit(model.X_train,
                                                                                                 model.y_train)
            theoretical_models.append(fitted)
        theoretical_score, theoretical_report = vote(theoretical_models, model.X_validate)
    print('')

    # OUTPUT -----------------------------------------------------------
    def human_readable_genes(genes_index):
        p = []
        try:
            with open('models/models_list.json', 'r') as fi:
                pp = json.load(fi)
        except FileNotFoundError as ex:
            print('\033[93m' + "Couldn't convert genes' names. Error: " + str(ex) +
                  "\nPrinting genes' indexes" + '\033[0m')
            p = genes_index
        else:
            for it in genes_index:
                p.append(pp[str(inst.models_index[it])])
        return p
    genes_list = 'Chosen classifiers:\n'
    for i in human_readable_genes(conv_genes(population.bestInGen.genes)):
        genes_list += i + '\n'

    print("Classifiers' indexes:", conv_genes(population.bestInGen.genes))
    print(genes_list)
    print("Score:", score, "in", population.genNo, "iterations")
    print(report)
    print("Separate scores:", separated_scores[:5])
    print("                ", separated_scores[5:])
    write_to_json("classifiers_scores", population.genFitness)
    if Model.TEST:
        print("Theoretical (assume: first 10 are best):", theoretical_score)
    else:
        print("Random committee (for comparison):", theoretical_score)
    print(theoretical_report)

    plot_scores_progress()
    plot_genes_in_last_gen()
    plot_avg_max_distance_progress()
    plot_best_phenotype_genes_progress()

    out_file_name = f'{dataset[9:]}-{col}-{metrics}-{pop}-{comm}-{Model.RUN_ID}'
    out_content = f'{conv_genes(population.bestInGen.genes)}\n' \
                  f'Score: {score} in {population.genNo} iterations\n' \
                  f'{report}\n' \
                  f'Separate scores: {separated_scores}\n' \
                  f'Random committee (for comparison): {theoretical_score}' \
                  f'\n{theoretical_report}\n\n ------------------------------------------ \n' \
                    f'{genes_list}'

    with open(f'output_files/plots/{Model.RUN_ID}/{out_file_name}.txt', 'w') as f:
        f.write(out_content)

    print('\033[94m' + f'All processes finished. Results may be found in output_files/plots/' + '\033[0m' +
          '\033[1m' + f'{Model.RUN_ID}' + '\033[0m')
