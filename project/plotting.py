import glob
import json
import os
import sys

import numpy as np

import matplotlib.pyplot as plt


def plot_scores_progress():
    name = 'classifiers_scores'
    list_of_files = glob.glob(f'output_files/{name}/*.json')
    file = max(list_of_files, key=os.path.getctime)
    try:
        with open(file, "r") as f:
            scores = json.load(f)
    except FileNotFoundError as e:
        print("An error occurred while creating graphs. Error:", e)
        sys.exit(2)
    x = np.array([i for i in range(len(scores))])
    y = np.array(scores)
    plt.plot(x, y, label="fitness scores")
    plt.legend()
    plt.savefig(f"output_files/plots/plot-for-{name}-{file[32:len(file) - 5]}")
    plt.show()


def plot_best_phenotype_genes_progress():
    name = 'gen_stats'
    list_of_files = glob.glob(f'output_files/{name}/*.json')
    file = max(list_of_files, key=os.path.getctime)
    try:
        with open(file, "r") as f:
            scores = json.load(f)
    except FileNotFoundError as e:
        print("An error occurred while creating graphs. Error:", e)
        sys.exit(2)

    x = np.array([i for i in range(len(scores))])
    y = np.array(scores)
    plt.scatter(x, y, label="genes")
    plt.legend()
    plt.savefig(f"output_files/plots/plot-for-{name}-{file[23:len(file) - 5]}")
    plt.show()


def plot_genes_in_last_gen():
    # population_dump
    pass


def plot_avg_max_distance_progress():
    # validation_res
    pass
