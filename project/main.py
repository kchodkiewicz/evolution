from Population import Population

if __name__ == '__main__':
    population = Population(10, 2, 10)

    print("PreCross")
    print(population.phenotypes[1].genes)
    print(population.phenotypes[2].genes)
    population.cross(population.phenotypes[1], population.phenotypes[2])
    print("PostCross")
    print(population.phenotypes[1].genes)
    print(population.phenotypes[2].genes)
    #if population.classification_did_finish():
        # natural selection
       # pass
    #else:
        # run classifiers
        #pass

