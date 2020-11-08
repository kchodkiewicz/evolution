import csv
import pandas as pd


# Handler for file opening etc.
class Model(object):
    dataset = None

    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        # TODO add file content to dataset variable



