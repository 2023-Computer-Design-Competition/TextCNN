import sklearn
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

dataset = pd.read_csv("my_data\\weibo_senti_100k.csv")
# print(dataset.head())
dataset = shuffle(dataset, ) 
# print(dataset.head())

dataset_train = dataset[:7000]

dataset_dev = dataset[7000:9000]

dataset_test = dataset[10000:15000]
# print(dataset_train.tail())
# print(dataset_dev.head())

dataset_train.to_csv("my_data\\my_train.csv")
dataset_dev.to_csv("my_data\\my_dev.csv")
dataset_test.to_csv("my_data\\test.csv")