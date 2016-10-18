from __future__ import print_function
import tensorflow as tf
import argparse

import csv_helpers

dataset = ["./data-set/iris_training.csv"]

file_length = csv_helpers.fileLen(dataset[0])

print(file_length)




