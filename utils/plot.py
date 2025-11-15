import matplotlib.pyplot as plt
import numpy as np
from supertree import SuperTree

def plot_tree(predictor):
    plt = SuperTree(predictor.model, predictor.X)