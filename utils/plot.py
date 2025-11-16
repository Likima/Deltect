import matplotlib.pyplot as plt
import numpy as np
from supertree import SuperTree
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Will have to play around with this to show more meaningful information
def plot_tree(predictor, html_path="tree.html"):
    st = SuperTree(predictor.model, predictor.X, predictor.y)
    st.save_html("tree")
    print("Tree saved. open in browser to view tree")


def visualize_results(predictor, results):
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    threshold = predictor.threshold

    y_test_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_binary,y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["Benign", "Pathogenic"])
    #plt.figure(figsize=(5,5))
    disp.plot(cmap="Blues", colorbar = False)
    plt.title("Confusion Matrix")
    plt.show()

    # Predicted vs. Actual Probabilities
    plt.figure(figsize=(6,5))
    plt.scatter(y_test,y_pred, alpha = 0.6)
    plt.xlabel("Actual Pathogenic Probability")
    plt.ylabel("Predicted Pathogenic Probability")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.show()

    # Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6,5))
    plt.scatter(y_pred, residuals, alpha = 0.6)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()

    # 4. Distribution of predicted prs
    plt.figure(figsize = (6,4))
    plt.hist(y_pred, bins = 20)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid(True)
    plt.show()

    # ROC Curve 
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


