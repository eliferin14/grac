import numpy as np
import matplotlib.pytplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default="processing_times.csv")
args = parser.parse_args()
print(args)