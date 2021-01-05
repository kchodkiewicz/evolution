# Tools -- its me and its good
import getopt
import json
import os
import sys
import time
from json import JSONEncoder

import numpy
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report

from models.Model import Model, calcScore


# Validate if fitness of last 20 classifiers is getting better
def fitness_is_progressing(fitness_scores_arr):
    if len(fitness_scores_arr) >= 20:
        score_sum = sum(fitness_scores_arr[len(fitness_scores_arr) - 20:])
        score_avg = score_sum / 20
        score_max = sorted(fitness_scores_arr, reverse=True)[0]
        if abs(score_avg - score_max) < 0.0001:
            return False
    return True


# Parse user input
def parse_args(argv):
    dataset_name = ''
    col_name = ''
    metrics_method = 'f1_score'
    pop_size = 1000
    committee_len = 10
    load_f = None
    verbose = False
    test = False
    pre_trained = False
    try:
        opts, args = getopt.getopt(argv, "hi:c:m:p:s:l:vtz", ["help", "if=", "column=", "metrics=", "pop_size=",
                                                              "committee_size=", "load_genes=", "verbose", "test",
                                                              "pre-trained"])
    except getopt.GetoptError as e:
        print('evo.py -i <infile.csv> -c <column> [-m <metrics> -p <population_size> -s <committee_size> -l '
              '<genes_file.json> -v -t -z]')
        print('\033[93m' + str(e) + '\033[0m')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('Usage:')
            print('evo.py -i <infile.csv> -c <column> [-m <metrics> -p <population_size> -s <committee_size> -l '
                  '<genes_file.json> -v -t -z]')
            print('Args:')
            print("-i, --if= - path and name of .csv file containing dataset, REQUIRED")
            print("-c, --column= - name of column containing correct classes, REQUIRED")
            print("-m, --metrics= - metrics method used for evaluating performance [accuracy_score, f1_score], "
                  "default : f1_score")
            print("-p, --pop_size= - size of population, default : 1000")
            print("-s, --committee_size= - size of initial committee, default 10")
            print("-l, --load_genes= - path and name of .json file containing genes")
            print("-v, --verbose - show progress info")
            print("-t, --test - use testing list of classifiers (assumption: first 10 are best)")
            print("-z, --pre-trained - use pre trained set of classifiers "
                  "(set is available at /models/pre_trained_classifiers)")
            sys.exit(0)
        elif opt in ("-i", "--if"):
            try:
                with open(arg):
                    dataset_name = arg
            except FileNotFoundError as e:
                print('\033[93m' + str(e) + '\033[0m')
        elif opt in ("-c", "--column"):
            col_name = arg
        elif opt in ("-m", "--metrics"):
            if arg == "accuracy_score" or arg == "f1_score":
                metrics_method = arg
            else:
                if metrics_method != 'accuracy_score':
                    print('\033[93m' + 'Incorrect metrics method: ' + str(arg) + '\033[0m')
                    sys.exit(2)
        elif opt in ("-p", "--pop_size"):
            if not arg.isnumeric():
                print('\033[93m' + "Incorrect population size: " + str(arg) + '\033[0m')
                sys.exit(2)
            else:
                try:
                    tmp_p = int(arg)
                    if tmp_p < 0:
                        raise ValueError
                except ValueError:
                    print('\033[93m' + "Population size must be a positive integer. Provided: " + str(arg) + '\033[0m')
                else:
                    pop_size = tmp_p
        elif opt in ("-s", "--committee_size"):
            if not arg.isnumeric():
                print('\033[93m' + "Incorrect committee size. Provided: " + str(arg) + '\033[0m')
                sys.exit(2)
            else:
                try:
                    tmp_c = int(arg)
                    if tmp_c < 0:
                        raise ValueError
                except ValueError:
                    print('\033[93m' + "Committee size must be a positive integer. Provided: " + str(arg) + '\033[0m')
                else:
                    committee_len = tmp_c
        elif opt in ("-l", "--load_genes"):
            try:
                with open(arg):
                    load_f = arg
            except FileNotFoundError as e:
                print('\033[93m' + str(e) + '\033[0m')
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-t", "--test"):
            test = True
        elif opt in ("-z", "--pre-trained"):
            pre_trained = True

    if dataset_name == '' or col_name == '':
        print('\033[93m' + 'Dataset and column name are required' + '\033[0m')
        sys.exit(2)
    return dataset_name, col_name, metrics_method, pop_size, committee_len, load_f, verbose, test, pre_trained


# Remove columns with same data (low variance)
def variance_threshold_selector(data, threshold=0.9):
    try:
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]
    except ValueError:
        return data


def clear_cache():
    paths = ['output_files/classifiers_scores/', 'output_files/gen_stats/', 'output_files/validation_res/']
    # 'models/trained_classifiers/', 'models/vanilla_classifiers/']
    for path in paths:
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    if not name.startswith(os.path.join(root, 'plots/')):
                        os.rmdir(os.path.join(root, name))
        except OSError as e:
            print('\033[93m' + str(e) + '\033[0m')
        print('Deleting directories at ' + str(path))
    try:
        os.mkdir('output_files/gen_stats', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('output_files/plots', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('output_files/classifiers_scores', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('output_files/population_dump', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('output_files/validation_res', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('models/vanilla_classifiers', 0o777)
    except FileExistsError:
        pass
    try:
        os.mkdir('models/trained_classifiers', 0o777)
    except FileExistsError:
        pass


def create_dir(path, run_id):
    dir_path = os.path.join(f'output_files/{path}', str(run_id))
    try:
        os.mkdir(dir_path, 0o777)
    except FileExistsError:
        pass
    finally:
        return dir_path


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Write to specified .json file
def write_to_json(path, content):
    if Model.RUN_ID is None:
        print('\033[93m' + "An error occurred while writing to a file. Aborting" + '\033[0m')
        sys.exit(2)
    dir_path = os.path.join(f'output_files/{path}', str(Model.RUN_ID))
    try:
        os.mkdir(dir_path, 0o777)
    except FileExistsError:
        pass
    finally:
        try:
            with open(f'{dir_path}/{time.localtime()[0]}-{time.localtime()[1]}-'
                      f'{time.localtime()[2]}_'
                      f'{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}.json', 'w') as filename:
                json.dump(content, filename, indent=4, cls=NumpyArrayEncoder)
        except Exception as e:
            print('\033[93m' + str(e) + '\033[0m')


# Return indexes of true genes
def conv_genes(genes_bool):
    arr = []
    for i, gen in enumerate(genes_bool):
        if gen:
            arr.append(i)
    return arr


# Make predictions for selected group of classifiers
def predictSelected(classifiers_arr, X):
    res = []
    for i, instance in enumerate(classifiers_arr):
        predictions = instance.predict(X)
        res.append(predictions)
    return res


# Print progress bar
def print_progress(i, end, msg):
    if Model.VERBOSE:
        bar_len = 10
        block = int(round((i / end) * bar_len))
        if i != end:
            dots = ('.' * (i % 3 + 1))
        else:
            dots = ' [DONE]'

        if i == 0:
            print('{:<15}'.format(msg), '[ ', '#' * block, ' ' * int(bar_len - block), ']', i, '/', str(end) + ' ' +
                  dots, end='',
                  flush=True)
        print('\r' + '{:<15}'.format(msg), '[ ', '#' * block, ' ' * int(bar_len - block), ']', i, '/', str(end) + ' ' +
              dots, end='',
              flush=True)


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
    report = classification_report(Model.y_validate, committee_answers)
    # calculate score of committee
    return calcScore(committee_answers, verify=True), report
