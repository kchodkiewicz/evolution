from Population import Population

if __name__ == '__main__':
    population = Population(10, 10)

    population.cross(population.phenotypes[1], population.phenotypes[2])
    x = 1
    y = x
    #if population.classification_did_finish():
        # natural selection
       # pass
    #else:
        # run classifiers
        #pass

