import getopt
import json
import sys
import time
from sklearn.feature_selection import VarianceThreshold

from models.Model import Model


def fitness_is_progressing(fitness_scores_arr):
    if len(fitness_scores_arr) >= 20:
        score_sum = sum(fitness_scores_arr[len(fitness_scores_arr) - 20:])
        score_avg = score_sum / 20
        score_max = sorted(fitness_scores_arr, reverse=True)[0]
        if abs(score_avg - score_max) < 0.0001:
            return False
    return True


def parse_args(argv):
    dataset_name = ''
    col_name = ''
    metrics_method = 'accuracy_score'
    pop_size = 1000
    committee_len = 10
    load_f = None
    try:
        opts, args = getopt.getopt(argv, "hi:c:m:p:s:l:", ["help", "if=", "column=", "metrics=", "pop_size=",
                                                           "committee_size=", "load_genes="])
    except getopt.GetoptError as e:
        print('evo.py -i <infile.csv> -c <column> [-m <metrics> -p <population_size> -s <committee_size> -l '
              '<genes_file.json>]')
        print(e)
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            print('evo.py -i <infile> -c <column> -m <metrics> -p <population_size> -s <committee_size>')
            print("-i, --if= - path and name of .csv file containing dataset, REQUIRED")
            print("-c, --column= - name of column containing correct classes, REQUIRED")
            print("-m, --metrics= - metrics method used for evaluating performance [accuracy_score, f1_score], "
                  "default : accuracy_score")
            print("-p, --pop_size= - size of population, default : 1000")
            print("-s, --committee_size= - size of initial committee, default 10")
            print("-l, --load_genes= - path and name of .json file containing genes")
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
        elif opt in ("-l", "--load_genes"):
            try:
                with open(arg) as f:
                    load_f = arg
            except FileNotFoundError as e:
                print(e)

    if dataset_name == '' or col_name == '':
        print('Dataset and column name are required')
        sys.exit(2)
    return dataset_name, col_name, metrics_method, pop_size, committee_len, load_f


def variance_threshold_selector(data, threshold=0.9):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def write_to_json(path, content):
    try:
        with open(f"output_files/{path}/timed-{time.localtime()[0]}-{time.localtime()[1]}-"
                  f"{time.localtime()[2]}_"
                  f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}.json", "w") as filename:
            json.dump(content, filename, indent=4)
    except Exception as e:
        print(e)


# Return indexes of true genes
def conv_genes(genes_bool):
    arr = []
    for i, gen in enumerate(genes_bool):
        if gen:
            arr.append(i)
    return arr


def predictSelected(classifiers_arr, X):
    res = []
    for i, instance in enumerate(classifiers_arr):
        predictions = instance.predict(X)
        res.append(predictions)
    return res


def print_progress(i, end, msg):
    if i == 0:
        print(msg, i, "/", end, end='', flush=True)
    print("\r", msg, i, "/", end, end='', flush=True)


def vote(models_arr, X):
    # make predictions for final models
    final_predicts = predictSelected(models_arr, X)
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
    return model.calcScore(predictions=committee_answers, verify=True)
