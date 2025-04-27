from datasets import load_dataset
import json
import os
import torch
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, dataset_name, dataset_type):
        if dataset_type == "local":
            if not os.path.exists(dataset_name):
                raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
            if "ECHR" in dataset_name:
                self.data = []
                for root, dirs, files in os.walk(dataset_name):
                    files.sort()
                    for i, file in enumerate(files):
                        with open(os.path.join(dataset_name, file), 'r') as f:
                            d = json.load(f)
                        self.data.append(d["CONCLUSION"])
            else:
                f = open(dataset_name, 'r')
                self.data = json.load(f)
                f.close()
        elif dataset_type == "datasets":
            self.data = []
            data_hug = load_dataset(dataset_name)
            for key in data_hug.keys():
                print("dataset size: {}".format(len(data_hug[key])))
                if 'output' in data_hug[key].features:
                    for item in data_hug[key]['output']:
                        self.data.append(item)
        elif dataset_type == "github":
            if "skytrax-reviews-dataset" in dataset_name:
                if not os.path.exists(dataset_name):
                    raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
                f = open(dataset_name, 'r')
                self.data = pd.read_csv(dataset_name)['content']
                f.close()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
                    
    def get_data(self):
        return self.data