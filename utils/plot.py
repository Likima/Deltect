import matplotlib.pyplot as plt
import numpy as np
from supertree import SuperTree

def plot_tree(predictor, html_path="tree.html"):
    st = SuperTree(predictor.model, predictor.X, predictor.y)
    st.save_html("tree")
    print("Tree saved. open in browser to view tree")