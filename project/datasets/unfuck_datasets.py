import pandas as pd

dataset = pd.read_csv("avocado.csv")


new = dataset[2:]

pd.write_csv(new, "avocado2.csv")
