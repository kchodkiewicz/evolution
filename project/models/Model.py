import csv


class Model(object):
    dataset = []

    def __init__(self):
        with open("dataset/file.csv") as f:
            self.dataset = []
            # TODO add file content to dataset variable
