from Utilscww_fr_vit import xlsread, plotMFs, most_frequent, compute_words_confidence
from Translationcww_fr_vit import translationcww_for_vit
from Manipulationcww_fr_vit import manipulationcww_for_vit
from Retranslationcww_fr_vit import EucledianSimilarity
import numpy as np

def process_labels_confidence(label, conf):
    # To find out the most frequent class label
    max_label = most_frequent(label)

    # Find linguistic term set & confidence of each feature
    words, conf = compute_words_confidence(label, max_label, conf)

    # Translation
    CentroidsMFs = np.zeros((len(words)), dtype=float)
    MFs = np.zeros((len(words), 3), dtype=float)
    MFs, CentroidsMFs = translationcww_for_vit(5, 0, 1, words)
    plotMFs(words, MFs, "FOUDataPlot.png")

    # Manipulation
    YLWA = manipulationcww_for_vit(conf, words, MFs)
    tempYLWA = np.array(YLWA).reshape((1, 3))
    plotMFs([r'$Y_{LWA}$'], tempYLWA, "YLWAPlot.png")

    # Retranslation
    S, indices, decode = EucledianSimilarity(words, YLWA, MFs)

    # Printing the results
    print("labels of eyes, nose, lips: ", label)
    print("confidence scores of eyes, nose, lips: ", conf)
    print("Max Label: ", max_label)
    print("Term set: ", words)
    print("MFs of term set: ", MFs)
    print("MF centroids of term set: ", CentroidsMFs)
    print("YLWA", YLWA)
    print("Similarity matrix: ", S)
    print("Indices of similarity terms: ", indices)
    print("Most similar linguistic term is: ", decode)

    return decode  # Return final decision

if __name__ == "__main__":
    print("Run this script from test_main.py using the process_labels_confidence() function.")
