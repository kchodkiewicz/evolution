# Evo
> Evolutionary algorithm choosing best committee of classifiers

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Usage](#usage)
* [Code Examples](#code-examples)
* [Features](#features)
* [Contact](#contact)

## General info
Committee consists of group of different classifiers. Classifiers have different accuracy in varied types of datasets, so choosing best is crucial for effective predictions. Committee in general increases accuracy compared to separate usage and decreases overfitting.
Genetic algorithm is used for finding best committee of classifiers for specified dataset.

## Technologies
* Python3
* Scikit-learn
* Pandas
* Matplotlib
* UCI Machine Learning Repository
* Kaggle

## Setup

### Installation:


### Usage:

`python3 evo.py -i <infile> -c <column> -m <metrics> -p <population_size> -s <committee_size>`
* `-i, --if= - path and name of .csv file containing dataset, REQUIRED`
* `-c, --column= - name of column containing correct classes, REQUIRED`
* `-m, --metrics= - metrics method used for evaluating performance [accuracy_score, f1_score], default : accuracy_score`
* `-p, --pop_size= - size of population, default : 1000`
* `-s, --committee_size= - size of initial committee, default 10`
* `-l, --load_genes= - path and name of .json file containing genes`
* `-v, --verbose - show progress info`
* `-t, --test - use testing list of classifiers (assumption: first 10 are best)`

## Code Examples
`python3 evo.py -i dataset.csv -c Class -p 100 -s 10 -v`

## Features
List of features ready:
* Setting desired population size
* Setting desired committee size. Committe size may change if increase of accuracy of committee with different size is much better
* Loading pre-chosen population from .json file
* Finding classifier
* Visualising results of evolution

To-do list:
* Soft voting

## Contact
Created by [kchod.98@gmail.com] - feel free to contact me!
