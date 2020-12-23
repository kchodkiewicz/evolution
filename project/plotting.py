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
        ax.scatter(x, y, label='populacja ' + str(i), marker='x')  # Plot some data on the axes.

    ax.set_xlabel('Numer populacji')
    ax.set_ylabel('Numer klasyfikatora')
    ax.legend()  # Add a legend.
    ax.set_title('Rozkład genów najlepszych osobników w populacjach')

    draw(name)


def plot_avg_max_distance_progress():
    # validation_res
    pass
