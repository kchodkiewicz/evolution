from Population import Population

if __name__ == '__main__':
    population = Population(size=10, committee=10, gen_length=100)

    # population.test()
    print("PreCross")
    print(population.phenotypes[1].genes)
    print(population.phenotypes[2].genes)
    ch1, ch2 = population.cross(population.phenotypes[1], population.phenotypes[2])
    print("PostCross")
    print(ch1.genes)
    print(ch2.genes)

    if population.classification_did_finish():
        population.select()
        population.validate()
    else:
        population.run()

