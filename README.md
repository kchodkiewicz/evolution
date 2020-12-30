# evolution
evolutionary algorithm choosing from group of classifiers

Committee consists of group of different classifiers.
Genetic algorithm used for finding best committee of classifiers for specified dataset. 

Usage:
evo.py -i <infile> -c <column> -m <metrics> -p <population_size> -s <committee_size>
* -i, --if= - path and name of .csv file containing dataset, REQUIRED
* -c, --column= - name of column containing correct classes, REQUIRED
* -m, --metrics= - metrics method used for evaluating performance [accuracy_score, f1_score], default : accuracy_score
* -p, --pop_size= - size of population, default : 1000
* -s, --committee_size= - size of initial committee, default 10
* -l, --load_genes= - path and name of .json file containing genes
* -v, --verbose - show progress info
* -t, --test - use testing list of classifiers (assumption: first 10 are best)
