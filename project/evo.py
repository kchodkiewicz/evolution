# Main
import statistics
import sys
import time
from operator import xor
from random import shuffle, random, randint, uniform
from re import search
import matplotlib.pyplot as plt
import keyboard
import pandas as pd
from pandas import errors
from sklearn.model_selection import train_test_split
from Population import Population, conv_genes, write_to_json
from models.Model import Model
from models.instances import Instances
from utils import parse_args, variance_threshold_selector, fitness_is_progressing, predictSelected, vote, clear_outs, \
    create_dir
from plotting import plot_scores_progress, plot_best_phenotype_genes_progress, plot_genes_in_last_gen, \
    plot_avg_max_distance_progress
import numpy as np

if __name__ == '__main__':
    dataset, col, metrics, pop, comm, load_file, verbose, testing = parse_args(sys.argv[1:])

    # TESTING GROUND ---------------------------------------------------------------------------------------------------
    # test_inst = Instances()
    # test_inst.trainClassifiers(model.X_train, model.y_train)
    # test_inst.predictClassifiers(model.X_test)

    tab_start = """
    \\begin{table}[h!] \\label{tab:mut:test}
    \\begin{center}
    \\begin{tabular}{l l l l l}
    \\textbf{średnia arytmetyczna} & \\textbf{mediana} & \\textbf{dominanta}\\\\
    \\hline
        """
    tab_end = """
    \\end{tabular}
    \\caption{Wynik testu sprawdzającego poprawność funkcji mutującej.}
    \\end{center}
    \\end{table}
        """
    test_pop = Population(10000, 10, 40)
    # arr = []
    # for i, phenotype in enumerate(test_pop.phenotypes):
    #     phenotype.fitness = random()
    # sum_p = 0
    # for i in test_pop.phenotypes:
    #     sum_p += i.fitness
    # for phenotype in test_pop.phenotypes:
    #     phenotype.normalizedFitness = phenotype.fitness / sum_p
    #     # print(phenotype.normalizedFitness)
    # np.random.shuffle(test_pop.phenotypes)
    # test_count = 10000
    # for i in range(test_count):
    #     arr.append(test_pop.tournament_selection())
    #
    # # print(tab_start)
    # arr.sort(key=lambda p: p.phenotype_id)
    # arr = list(dict.fromkeys(arr))
    # verifarr = []
    #
    # for elem in arr:
    #     j = 0
    #     cc = elem.normalizedFitness
    #     while cc < 1:
    #         cc = cc * 10
    #         j += 1
    #
    #     verifarr.append(abs(elem.normalizedFitness - elem.counter / test_count) > 10 ** (-j) * 10)
    #     print(abs(elem.normalizedFitness - elem.counter / test_count), elem.normalizedFitness, 10 ** (-j) * 2)
    # # for elem in arr:
    # #     print(str(elem.phenotype_id) + ' & ' + str(elem.counter) + ' & ' + str(elem.counter / test_count) + ' & ' +
    # #           '{:04f}'.format(elem.normalizedFitness) + ' \\\\')
    # i = 0
    # for elem in verifarr:
    #     if elem:
    #         i += 1
    # print(i)
    # # print(tab_end)
    # par1 = test_pop.phenotypes[0]
    # par2 = test_pop.phenotypes[1]
    # par1.genes = [True for i in range(len(par1.genes))]
    #
    # i = 0
    # print(par1.genes, par2.genes)
    # ch1, ch2, cut1, cut2 = test_pop.cross(i, par1, par2)
    # print(cut1, cut2)
    # print(ch1.genes, ch2.genes)

    # print('\nMutate')
    # print(tab_start)
    # for _ in range(10):
    #     prev = []
    #     new = []
    #     for i in test_pop.phenotypes:
    #         prev.append(i.genes.copy())
    #         test_pop.mutate(i)
    #         new.append(i.genes.copy())
    #     tabs = []
    #     for i in range(len(prev)):
    #         itt = 0
    #         for j in range(len(prev[i])):
    #             if xor(prev[i][j], new[i][j]):
    #                 itt += 1
    #         tabs.append(itt)
    #
    #     avg = sum(tabs) / len(tabs)
    #     median = statistics.median(tabs)
    #     mode = statistics.mode(tabs)
    #     print(avg, '&', median, '&', mode, '\\\\')
    # print(tab_end)
    # par2.genes = [False for i in range(len(par2.genes))]

    # initialCommittee = 10
    # xp = [(uniform(0, 1), 10) for _ in range(1000)]
    # # xp = [(0.85, i) for i in range(84)]
    # # xp = []
    # # for i in range(1000):
    # #     for j in range(84):
    # #         if i % 100 == 0:
    # #             t = j
    # #         else:
    # #             t = 0
    # #         xp.append((i / 1000, t))
    # xp.sort()
    #
    # def punish_length(xd):
    #     dist = xd - initialCommittee
    #     yd = -(dist / (initialCommittee / 2)) ** 4 + 0.5
    #     if yd < 0:
    #         yd = 0
    #     return yd
    # fitnesses = []
    # for i in xp:
    #     fit = 0.8 * pow(i[0] + 1, 2) + 0.2 * punish_length(i[1])
    #     fitnesses.append(fit)
    #
    # fig, ax = plt.subplots()
    # x = np.array([xp[i][0] for i in range(len(fitnesses))])
    # y = np.array(fitnesses)
    # ax.plot(x, y, label="fitness scores")
    # ax.grid()
    # ax.set_xlabel('Skuteczność klasyfikacji')
    # ax.set_ylabel('Współczynnik przystosowania')
    # #  ax.set_title('Zależność współczynnika przystosowania od liczebności komitetu klasyfikatorów')
    #
    # try:
    #     plt.savefig(f"output_files/test.png", dpi=800)
    # except FileNotFoundError as e:
    #     print('\033[93m' + str(e) + '\033[0m')
    #     sys.exit(2)
    # except ValueError as e:
    #     print('\033[93m' + str(e) + '\033[0m')
    #     sys.exit(2)
    # sys.exit(1)

    # END TESTING GROUND -----------------------------------------------------------------------------------------------

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
    # TODO add tournament to dev branch
    fitness_scores = []
    while True:

        population.run_normally()
        # population.run_async(4)
        population.validate()
        population.select()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing(fitness_scores):
            break
        # if keyboard.is_pressed('q'):  # if key 'q' is pressed
        #     print('You Pressed A Key!')
        #     break  # finishing the loop

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
    if Model.TEST:
        print("Theoretical (assume: first 10 are best):", theoretical_score)

    plot_scores_progress()
    plot_genes_in_last_gen()
    plot_avg_max_distance_progress()
    plot_best_phenotype_genes_progress()
    print('\033[94m' + f'All processes finished. Results may be found in output_files/plots/' + '\033[0m' +
          '\033[1m' + f'{Model.RUN_ID}' + '\033[0m')
