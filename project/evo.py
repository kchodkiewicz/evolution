# Main
import sys
import time

import keyboard
import pandas as pd
from pandas import errors
from sklearn.model_selection import train_test_split
from Population import Population, conv_genes, write_to_json
from models.Model import Model
from models.instances import Instances
from utils import parse_args, variance_threshold_selector, fitness_is_progressing, predictSelected, vote, clear_outs
from plotting import plot_scores_progress, plot_best_phenotype_genes_progress, plot_genes_in_last_gen, \
    plot_avg_max_distance_progress


if __name__ == '__main__':
    dataset, col, metrics, pop, comm, load_file, verbose, testing = parse_args(sys.argv[1:])

    inst = Instances()
    model = Model()

    Model.verbose = verbose
    # Add id for run
    timestamp = time.time()
    Model.RUN_ID = str(timestamp).replace('.', '-')
    print('Run identifier:', Model.RUN_ID)
    Model.TEST = testing
    clear_outs()

    try:
        Model.dataset = pd.read_csv(dataset)
    except pd.errors.ParserError:
        print('\033[93m' + "Incorrect path to file" + '\033[0m')
        sys.exit(2)
    Model.METRICS_METHOD = metrics

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
    """
    # DATASET CLEARING
    dataset = "dataset-har-PUC-Rio-ugulino.csv"
    col = "class"
    Model.dataset = pd.read_csv(dataset, sep=';', decimal=',')
    X = Model.dataset.drop(columns=col)
    X['gender'] = X['gender'].map({'Woman': 1, 'Man': 0})
    X = X.drop(columns="user")
    y = Model.dataset[col]
    """
    Model.X_train, X_in, Model.y_train, y_in = train_test_split(X, y, test_size=0.3)
    Model.X_test, Model.X_validate, Model.y_test, Model.y_validate = train_test_split(X_in, y_in, test_size=0.3)

    print("Running for configuration:", "\n* dataset:", dataset, "\n* column:", col, "\n* metrics method:", metrics,
          "\n* population size:", pop, "\n* committee size:", comm, "\n* load_genes:", load_file)

    inst.trainClassifiers(model.X_train, model.y_train)
    inst.predictClassifiers(model.X_test)

    try:
        if comm > len(inst.trained_classifiers):
            raise ValueError
        population = Population(size=pop, committee=comm, gen_length=len(inst.predictions_arr))
    except ValueError:
        print('\033[93m' + "Committee size cannot be greater than amount of different classifiers (" +
              str(len(inst.predictions_arr)) + ")" + '\033[0m')
        sys.exit(2)

    if load_file is not None:
        try:
            population.load_population(load_file)
        except FileNotFoundError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')
            sys.exit(2)
        except ValueError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')
            sys.exit(2)
        except KeyError as e:
            print('\033[93m' + "Couldn't open genes file. Running default mode." + '\033[0m')
            sys.exit(2)

    # EVOLUTION --------------------------------------------------------

    fitness_scores = []
    while True:
        try:
            population.run_normally()
            # population.run_async(4)
            population.validate()
            population.select()
            fitness_scores.append(population.bestInGen.fitness)
            if not fitness_is_progressing(fitness_scores):
                break
            if keyboard.is_pressed('q'):  # if key 'q' is pressed
                print('You Pressed A Key!')
                break  # finishing the loop
        except KeyboardInterrupt:
            pass
        except KeyError:
            pass
        except ImportError:
            pass
    # SCORE OF EVOLVED MODELS ------------------------------------------
    # create final list of models
    final_models = []
    for i, gen in enumerate(population.bestInGen.genes):
        if gen:
            final_models.append(inst.trained_classifiers[i])
    score = vote(final_models, model.X_validate)

    # SEPARATE SCORES OF EVOLVED MODELS --------------------------------
    # create list of models used in final list
    separated_models = []
    for i, mod in enumerate(conv_genes(population.bestInGen.genes)):
        separated_models.append(inst.trained_classifiers[conv_genes(population.bestInGen.genes)[i]])
    # make predictions for every model
    separated_predicts = predictSelected(separated_models, model.X_validate)
    # calculate separate score for every model
    separated_scores = []
    for mod in separated_predicts:
        separated_scores.append(model.calcScore(mod, verify=True))

    # SCORE OF FIRST 10 MODELS -----------------------------------------
    # create theoretical list of models
    theoretical_models = []
    for i in range(10):
        theoretical_models.append(inst.trained_classifiers[i])
    theoretical_score = vote(theoretical_models, model.X_validate)

    # OUTPUT -----------------------------------------------------------
    def human_readable_genes(genes_index):
        p = []
        for it in genes_index:
            p.append(inst.trained_classifiers[it].__class__.__name__)
        return p

    print("Classifiers:", conv_genes(population.bestInGen.genes))
    print(human_readable_genes(conv_genes(population.bestInGen.genes)))
    print("Score:", score, "in", population.genNo, "iterations")
    print("Separate scores:", separated_scores[:5])
    print(separated_scores[5:])
    write_to_json("classifiers_scores", population.genFitness)
    print("Theoretical (assume: first 10 are best):", theoretical_score)

    plot_scores_progress()
    plot_genes_in_last_gen()
    plot_avg_max_distance_progress()
    plot_best_phenotype_genes_progress()
    print('\033[94m' + f'All processes finished. Results may be found in output_files/plots/'
                       f'\033[1m{Model.RUN_ID}\033[0m')
