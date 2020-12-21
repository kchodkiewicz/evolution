import sys, getopt

from sklearn.feature_selection import VarianceThreshold

from Population import Population, conv_genes, write_to_json
from models.Model import Model
import pandas as pd
from pandas import errors
from sklearn.model_selection import train_test_split
from models.instances import Instances, predictSelected


def fitness_is_progressing():
    if len(fitness_scores) >= 20:
        score_sum = sum(fitness_scores[len(fitness_scores) - 20:])
        score_avg = score_sum / 20
        score_max = sorted(fitness_scores, reverse=True)[0]
        if abs(score_avg - score_max) < 0.0001:
            return False
    return True


def parse_args(argv):
    dataset_name = ''
    col_name = ''
    metrics_method = 'accuracy_score'
    pop_size = 1000
    committee_len = 10
    try:
        opts, args = getopt.getopt(argv, "hi:c:m:p:s:", ["help", "if=", "column=", "metrics=", "pop_size=",
                                                         "committee_size="])
    except getopt.GetoptError as e:
        print('evo.py -i <infile> -c <column> -m <metrics> -p <population_size> -s <committee_size>')
        print(e)
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            print('evo.py -i <infile> -c <column> -m <metrics> -p <population_size> -s <committee_size>')
            print("-i, --if= - path and name of .csv file containing dataset, REQUIRED")
            print("-c, --column= - name of column containing correct classes, REQUIRED")
            print("-m - metrics method used for evaluating performance [accuracy_score, f1_score], default : "
                  "accuracy_score")
            print("-p - size of population, default : 1000")
            print("-s - size of initial committee, default 10")
            sys.exit()
        elif opt in ("-i", "--if"):
            try:
                with open(arg) as f:
                    dataset_name = arg
            except FileNotFoundError as e:
                print(e)
        elif opt in ("-c", "--column"):
            col_name = arg
        elif opt in ("-m", "--metrics"):
            if arg == "accuracy_score" or arg == "f1_score":
                metrics_method = arg
            else:
                print('Incorrect metrics method')
                sys.exit(2)
        elif opt in ("-p", "--pop_size"):
            if not arg.isnumeric():
                print("Incorrect population size")
                sys.exit(2)
            else:
                try:
                    tmp_p = int(arg)
                    if tmp_p < 0:
                        raise ValueError
                except ValueError:
                    print("Population size must be a positive integer")
                else:
                    pop_size = tmp_p
        elif opt in ("-s", "--committee_size"):
            if not arg.isnumeric():
                print("Incorrect committee size")
                sys.exit(2)
            else:
                try:
                    tmp_c = int(arg)
                    if tmp_c < 0:
                        raise ValueError
                except ValueError:
                    print("Committee size must be a positive integer")
                else:
                    committee_len = tmp_c

    if dataset_name == '' or col_name == '':
        print('Dataset and column name are required')
        sys.exit(2)
    return dataset_name, col_name, metrics_method, pop_size, committee_len


def variance_threshold_selector(data, threshold=0.9):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


if __name__ == '__main__':

    dataset, col, metrics, pop, comm = parse_args(sys.argv[1:])

    print("Running for configuration:", "\n* dataset:", dataset, "\n* column:", col, "\n* metrics method:", metrics,
          "\n* population size:", pop, "\n* committee size:", comm)

    try:
        Model.dataset = pd.read_csv(dataset)
    except pd.errors.ParserError:
        print("Incorrect path to file")
        sys.exit(2)
    Model.METRICS_METHOD = metrics

    # remove features with low variance (same value in 90% of samples)
    variance_threshold_selector(Model.dataset, 0.9)
    try:
        X = Model.dataset.drop(columns=col)
    except AttributeError:
        print("Column with name:", col, "not found in provided dataset")
        sys.exit(2)
    except KeyError:
        print("Column with name:", col, "not found in provided dataset")
        sys.exit(2)

    X = X.drop(columns="id", errors='ignore')
    X = X.drop(columns="Id", errors='ignore')
    X = X.drop(columns="ID", errors='ignore')
    X = X.drop(columns="Identification", errors='ignore')
    X = X.drop(columns="identification", errors='ignore')

    y = Model.dataset[col]
    Model.X_train, X_in, Model.y_train, y_in = train_test_split(X, y, test_size=0.3)
    Model.X_test, Model.X_validate, Model.y_test, Model.y_validate = train_test_split(X_in, y_in, test_size=0.3)
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
    try:
        if comm > len(inst.trained_classifiers):
            raise ValueError
        population = Population(size=pop, committee=comm, gen_length=len(inst.trained_classifiers))
    except ValueError:
        print("Committee size cannot be greater than amount of different classifiers (",
              len(inst.trained_classifiers), ")")
        sys.exit(2)
    fitness_scores = []
    while True:
        population.run_normally()
        # population.run_async(4)
        population.validate()
        population.select()
        fitness_scores.append(population.bestInGen.fitness)
        if not fitness_is_progressing():
            break

    # create final list of models
    final_models = []
    for i, gen in enumerate(population.bestInGen.genes):
        if gen:
            final_models.append(inst.trained_classifiers[i])
    # make predictions for final models
    final_predicts = predictSelected(final_models, Model.X_validate)
    # create list of answers for final committee
    committee_answers = []
    for i in range(len(final_predicts[0])):
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
    # calculate score of committee
    score = model.calcScore(predictions=committee_answers, verify=True)

    # create list of models used in final list
    separated_models = []
    for i, mod in enumerate(conv_genes(population.bestInGen.genes)):
        separated_models.append(inst.trained_classifiers[conv_genes(population.bestInGen.genes)[i]])
    # make predictions for every model
    separated_predicts = predictSelected(separated_models, Model.X_validate)
    # calculate separate score for every model
    separated_scores = []
    for mod in separated_predicts:
        separated_scores.append(model.calcScore(mod, verify=True))

    print("Classifiers:", conv_genes(population.bestInGen.genes))
    print("Score:", score)
    print("Separate scores:", separated_scores)
    write_to_json("classifiers_scores", population.genFitness)

    # TODO check adaboost
    # TODO VotingClassifier <- may be refactor ??
    # adaboost = AdaBoost()

    # create final list of models
    final_models = []
    for i in range(10):
        final_models.append(inst.trained_classifiers[i])
    # make predictions for final models
    final_predicts = predictSelected(final_models, Model.X_validate)
    # create list of answers for final committee
    committee_answers = []
    for i in range(len(final_predicts[0])):
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
    # calculate score of committee
    score = model.calcScore(predictions=committee_answers, verify=True)
    print("Teoretical:", score)
