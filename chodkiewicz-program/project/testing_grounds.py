# TESTING GROUND ---------------------------------------------------------------------------------------------------
# test_inst = Instances()
# test_inst.trainClassifiers(model.X_train, model.y_train)
# test_inst.predictClassifiers(model.X_test)

# tab_start = """
# \\begin{table}[h!] \\label{tab:mut:test}
# \\begin{center}
# \\begin{tabular}{l l l l l}
# \\textbf{średnia arytmetyczna} & \\textbf{mediana} & \\textbf{dominanta}\\\\
# \\hline
#     """
# tab_end = """
# \\end{tabular}
# \\caption{Wynik testu sprawdzającego poprawność funkcji mutującej.}
# \\end{center}
# \\end{table}
#     """
# test_pop = Population(10000, 10, 40)
# arr = []
# for i, phenotype in enumerate(test_pop.phenotypes):
#     phenotype.fitness = random()
# sum_p = 0
# for i in test_pop.phenotypes:
#     sum_p += i.fitness
# for phenotype in test_pop.phenotypes:
#     phenotype.normalizedFitness = phenotype.fitness / sum_p
#     # print(phenotype.normalizedFitness)
# np.random.shuffle(test_pop.phenotypes)
# test_count = 10000
# for i in range(test_count):
#     arr.append(test_pop.tournament_selection())
#
# # print(tab_start)
# arr.sort(key=lambda p: p.phenotype_id)
# arr = list(dict.fromkeys(arr))
# verifarr = []
#
# for elem in arr:
#     j = 0
#     cc = elem.normalizedFitness
#     while cc < 1:
#         cc = cc * 10
#         j += 1
#
#     verifarr.append(abs(elem.normalizedFitness - elem.counter / test_count) > 10 ** (-j) * 10)
#     print(abs(elem.normalizedFitness - elem.counter / test_count), elem.normalizedFitness, 10 ** (-j) * 2)
# # for elem in arr:
# #     print(str(elem.phenotype_id) + ' & ' + str(elem.counter) + ' & ' + str(elem.counter / test_count) + ' & ' +
# #           '{:04f}'.format(elem.normalizedFitness) + ' \\\\')
# i = 0
# for elem in verifarr:
#     if elem:
#         i += 1
# print(i)
# # print(tab_end)
# par1 = test_pop.phenotypes[0]
# par2 = test_pop.phenotypes[1]
# par1.genes = [True for i in range(len(par1.genes))]
#
# i = 0
# print(par1.genes, par2.genes)
# ch1, ch2, cut1, cut2 = test_pop.cross(i, par1, par2)
# print(cut1, cut2)
# print(ch1.genes, ch2.genes)

# print('\nMutate')
# print(tab_start)
# for _ in range(10):
#     prev = []
#     new = []
#     for i in test_pop.phenotypes:
#         prev.append(i.genes.copy())
#         test_pop.mutate(i)
#         new.append(i.genes.copy())
#     tabs = []
#     for i in range(len(prev)):
#         itt = 0
#         for j in range(len(prev[i])):
#             if xor(prev[i][j], new[i][j]):
#                 itt += 1
#         tabs.append(itt)
#
#     avg = sum(tabs) / len(tabs)
#     median = statistics.median(tabs)
#     mode = statistics.mode(tabs)
#     print(avg, '&', median, '&', mode, '\\\\')
# print(tab_end)
# par2.genes = [False for i in range(len(par2.genes))]

# initialCommittee = 10
# xp = [(uniform(0, 1), 10) for _ in range(1000)]
# # xp = [(0.85, i) for i in range(84)]
# # xp = []
# # for i in range(1000):
# #     for j in range(84):
# #         if i % 100 == 0:
# #             t = j
# #         else:
# #             t = 0
# #         xp.append((i / 1000, t))
# xp.sort()
#
# def punish_length(xd):
#     dist = xd - initialCommittee
#     yd = -(dist / (initialCommittee / 2)) ** 4 + 0.5
#     if yd < 0:
#         yd = 0
#     return yd
# fitnesses = []
# for i in xp:
#     fit = 0.8 * pow(i[0] + 1, 2) + 0.2 * punish_length(i[1])
#     fitnesses.append(fit)
#
# fig, ax = plt.subplots()
# x = np.array([xp[i][0] for i in range(len(fitnesses))])
# y = np.array(fitnesses)
# ax.plot(x, y, label="fitness scores")
# ax.grid()
# ax.set_xlabel('Skuteczność klasyfikacji')
# ax.set_ylabel('Współczynnik przystosowania')
# #  ax.set_title('Zależność współczynnika przystosowania od liczebności komitetu klasyfikatorów')
#
# try:
#     plt.savefig(f"output_files/test.png", dpi=800)
# except FileNotFoundError as e:
#     print('\033[93m' + str(e) + '\033[0m')
#     sys.exit(2)
# except ValueError as e:
#     print('\033[93m' + str(e) + '\033[0m')
#     sys.exit(2)
# sys.exit(1)

# END TESTING GROUND -----------------------------------------------------------------------------------------------
