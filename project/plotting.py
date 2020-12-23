import glob
import json
import os
import sys

import numpy as np

import matplotlib.pyplot as plt
from models.Model import Model


def draw(name):
    try:
        plt.savefig(f"output_files/plots/{Model.RUN_ID}-{name}")
    except FileNotFoundError as e:
        print('\033[93m' + str(e) + '\033[0m')
        sys.exit(2)
    except ValueError as e:
        print('\033[93m' + str(e) + '\033[0m')
        sys.exit(2)
    print('Drawing ' + name)
    plt.show()


def plot_scores_progress():
    name = 'classifiers_scores'
    list_of_dirs = glob.glob(f'output_files/{name}/*')
    curr_dir = max(list_of_dirs, key=os.path.getctime)
    list_of_files = glob.glob(f'{curr_dir}/*.json')
    file = max(list_of_files, key=os.path.getctime)
    fig, ax = plt.subplots()
    try:
        with open(file, "r") as f:
            scores = json.load(f)
    except FileNotFoundError as e:
        print("An error occurred while creating graphs. Error:", e)
        sys.exit(2)
    x = np.array([i for i in range(len(scores))])
    y = np.array(scores)
    ax.plot(x, y, label="fitness scores")
    ax.set_xlabel('Numer osobnika')
    ax.set_ylabel('Współczynnik przystosowania')
    ax.legend()
    ax.set_title('Wyniki poszczególnych osobników')
    draw(name)


def plot_best_phenotype_genes_progress():
    name = 'gen_stats'
    list_of_dirs = glob.glob(f'output_files/{name}/*')
    curr_dir = max(list_of_dirs, key=os.path.getctime)
    list_of_files = glob.glob(f'{curr_dir}/*.json')

    fig, ax = plt.subplots()  # Create a figure and an axes.
    for i, file in enumerate(list_of_files):
        keys = []
        values = []
        try:
            with open(file, "r") as f:
                scores = json.load(f)
        except FileNotFoundError as e:
            print("An error occurred while creating graphs. Error:", e)
            sys.exit(2)
        for k, v in scores.items():
            keys.append(k)
            values.append(scores[k])
        x = np.array([keys[i] for _ in range(len(values[i]))])
        y = np.array(values[i])
        ax.scatter(x, y, label='populacja ' + str(i), marker='x')  # Plot some data on the axes.

    ax.set_xlabel('Numer populacji')
    ax.set_ylabel('Numer klasyfikatora')
    ax.legend()  # Add a legend.
    ax.set_title('Rozkład genów najlepszych osobników w populacjach')

    draw(name)


def plot_genes_in_last_gen():
    """
    name = 'population_dump'
    list_of_dirs = glob.glob(f'output_files/{name}/*')
    curr_dir = max(list_of_dirs, key=os.path.getctime)
    list_of_files = glob.glob(f'{curr_dir}/*.json')

    fig, ax = plt.subplots()  # Create a figure and an axes.
    for i, file in enumerate(list_of_files):
        keys = []
        values = []
        try:
            with open(file, "r") as f:
                scores = json.load(f)
        except FileNotFoundError as e:
            print("An error occurred while creating graphs. Error:", e)
            sys.exit(2)
        for k, v in scores.items():
            keys.append(k)
            values.append(scores[k])
        x = np.array([keys[i] for _ in range(len(values[i]))])
        y = np.array(values[i])
        ax.scatter(x, y, label='osobnik ' + str(i), marker='x')  # Plot some data on the axes.

    ax.set_xlabel('Numer osobnika ostatniej generacji')
    ax.set_ylabel('Numer klasyfikatora')
    ax.legend()  # Add a legend.
    ax.set_title('Rozkład genów osobników w ostatniej populacji')

    draw(name)
    """
    name = 'population_dump'
    list_of_dirs = glob.glob(f'output_files/{name}/*')
    curr_dir = max(list_of_dirs, key=os.path.getctime)
    list_of_files = glob.glob(f'{curr_dir}/*.json')

    for i, file in enumerate(list_of_files):
        keys = []
        values = []
        try:
            with open(file, "r") as f:
                scores = json.load(f)
        except FileNotFoundError as e:
            print("An error occurred while creating graphs. Error:", e)
            sys.exit(2)
    labels = []
    men_means = 0
    women_means = 0
    for k, v in scores.items():
        labels.append(k)
        if v:
            men_means += 1
        else:
            women_means += 1
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
    rects2 = ax.bar(x + width / 2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def plot_avg_max_distance_progress():
    name = 'validation_res'
    list_of_dirs = glob.glob(f'output_files/{name}/*')
    curr_dir = max(list_of_dirs, key=os.path.getctime)
    list_of_files = glob.glob(f'{curr_dir}/*.json')
    file = max(list_of_files, key=os.path.getctime)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    #  for i, file in enumerate(list_of_files):
    keys = []
    values = []
    labels = []
    try:
        with open(file, "r") as f:
            scores = json.load(f)
    except FileNotFoundError as e:
        print("An error occurred while creating graphs. Error:", e)
        sys.exit(2)
    width = 10 / len(scores)
    for k, v in scores.items():
        keys.append(k)
        values.append(scores[k])
        labels.append(k)

    for k, v in scores.items():
        y_arr = [v['max'], v['avg'], abs(v['max'] - v['avg'])]
        ax.bar(k, y_arr[0], width, label='populacja ' + str(k) + ' max')
        ax.bar(k, y_arr[1], width, label='populacja ' + str(k) + ' avg')
        ax.bar(k, y_arr[2], width, label='populacja ' + str(k) + ' dystans')
        ax.scatter([k for _ in range(len(y_arr))], y_arr, label='populacja ' + str(k), marker='x')

    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Dystans max - avg')
    ax.legend()
    leg = ax.legend(loc=0, ncol=2, prop={'size': 8})
    leg.get_frame().set_alpha(0.4)
    ax.set_title('Wykres zbiegania się wyników osobników na przestrzeni populacji')

    draw(name)
