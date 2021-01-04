# Main
import json
import os
import pickle
import sys
import time
from random import randint
import pandas as pd
from pandas import errors
from sklearn.model_selection import train_test_split
from Population import Population, conv_genes, write_to_json
from models.Model import Model
from models.instances import Instances
from utils import parse_args, variance_threshold_selector, fitness_is_progressing, predictSelected, vote, clear_cache, \
    print_progress
from plotting import plot_scores_progress, plot_best_phenotype_genes_progress, plot_genes_in_last_gen, \
    plot_avg_max_distance_progress

if __name__ == '__main__':
    dataset, col, metrics, pop, comm, load_file, verbose, testing, pre_trained = parse_args(sys.argv[1:])

    # TESTING GROUND ---------------------------------------------------------------------------------------------------
    # test_inst = Instances()
    # test_inst.trainClassifiers(model.X_train, model.y_train)
    # test_inst.predictClassifiers(model.X_test)

    # tab_start = """
    # \\begin{table}[h!] \\label{tab:mut:test}
    # \\begin{center}
    # \\begin{tabular}{l l l l l}
    # \\textbf{średnia arytmetyczna} & \\textbf{mediana} & \\textbf{dominanta}\\\\
    # \\hline
    #     """
    # tab_end = """
    # \\end{tabular}
    # \\caption{Wynik testu sprawdzającego poprawność funkcji mutującej.}
    # \\end{center}
    # \\end{table}
    #     """
    # test_pop = Population(10000, 10, 40)
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

    if load_file is None:  # TODO new thingy not tested
        inst.trainClassifiers(model.X_train, model.y_train)
        inst.predictClassifiers(model.X_test)
    else:  # TODO new thingy not tested
        # TODO temporary bcos i fucked up
        # TODO save predictions_arr and load it
        tmp = [i for i in range(18 + 1)]
        for i in range(22, 25 + 1):
            tmp.append(i)
        for i in range(28, 51 + 1):
            tmp.append(i)
        for i in range(54, 83 + 1):
            tmp.append(i)
        inst.set_models_index_arr(tmp)
        inst.set_pred_arr([i for i in range(len(tmp))])

    try:
        if comm > len(inst.predictions_arr):
            raise ValueError
        population = Population(size=pop, committee=comm, gen_length=len(inst.predictions_arr))
    except ValueError:
        print('\033[93m' + "Committee size cannot be greater than the amount of different classifiers (" +
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
            with open(os.path.join('models/vanilla_classifiers', f'v-{inst.get_models_index(it)}.pkl'), 'rb') as fid:
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
            fitted = load_vanilla_classifier(randint(0, len(inst.predictions_arr) - 1)).fit(model.X_train,
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
                p.append(pp[str(inst.get_models_index(it))])
        return p


    print("Classifiers:", conv_genes(population.bestInGen.genes))
    print(human_readable_genes(conv_genes(population.bestInGen.genes)))
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
                  f'\n{theoretical_report}\n\n ------------------------------------------ \n'
    for i in human_readable_genes(conv_genes(population.bestInGen.genes)):
        out_content += i + '\n'
    with open(f'output_files/plots/{Model.RUN_ID}/{out_file_name}.txt', 'w') as f:
        f.write(out_content)

    print('\033[94m' + f'All processes finished. Results may be found in output_files/plots/' + '\033[0m' +
          '\033[1m' + f'{Model.RUN_ID}' + '\033[0m')
