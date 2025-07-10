from cProfile import label

from Utilscww_fr_vit import xlsread
from Utilscww_fr_vit import plotMFs
from Translationcww_fr_vit import translationcww_for_vit
from Manipulationcww_fr_vit import manipulationcww_for_vit
from Retranslationcww_fr_vit import EucledianSimilarity
from Utilscww_fr_vit import most_frequent
from Utilscww_fr_vit import compute_words_confidence
import pandas as pd
import numpy as np

if __name__ == "__main__":

    #Variable declaration
    label = ['', '', '']
    words = ['', '', '', '', '']
    conf = [0.0, 0.0, 0.0]

    #Label of each feature
    label[0] = input("Enter the label for eyes (autistic/non-autistic): ")
    label[1] = input("Enter the label for nose (autistic/non-autistic) ")
    label[2] = input("Enter the label for lips (autistic/non-autistic) ")

    # Confidence value of each feature
    conf[0] = float(input("Enter the confidence value for eyes: (without percentage): "))
    conf[1] = float(input("Enter the confidence value for nose: (without percentage): "))
    conf[2] = float(input("Enter the confidence value for lips: (without percentage): "))

    # To find out the most frequent class label
    max_label = most_frequent(label)

    # Find linguistic term set & confidence of each feature
    words,conf = compute_words_confidence(label, max_label, conf)

    # Translation
    CentroidsMFs = np.zeros((len(words)), dtype=float)
    MFs = np.zeros((len(words), 3), dtype=float)
    MFs, CentroidsMFs = translationcww_for_vit(5, 0, 1, words)
    #plotMFs(words, MFs, "FOUDataPlot.png")

    #Manipulation
    YLWA = manipulationcww_for_vit(conf, words, MFs)
    tempYLWA = np.array(YLWA)
    tempYLWA = tempYLWA.reshape((1, 3))
    #plotMFs([r'$Y_{LWA}$'], tempYLWA, "YLWAPlot.png")

    #Retranslation
    S = np.zeros((len(words)), dtype=float)
    S, indices, decode = EucledianSimilarity(words, YLWA, MFs)

    #Concatenation max label and linguistic term
    autism_dict = {
        "AV": "Very Low",
        "AL": "Low",
        "AM": "𝑀𝑜𝑑𝑒𝑟𝑎𝑡𝑒",
        "AH": "High",
        "AE": "Very High"
    }

    non_autism_dict = {
        "NE": "Very High",
        "NH": "High",
        "NM": "Moderate",
        "NL": "Low",
        "NL": "Very Low"
    }

    if max_label == "autistic":
        face_label = "autistic" + " " + autism_dict[decode[0]]
    else:
        face_label = "non-autistic" + " " + non_autism_dict[decode[0]]

    #Printing the results
    """print("labels of eyes, nose, lips: ", label)
    print("confidence scores of eyes, nose, lips: ",conf)
    print("Max Label: ", max_label)
    print("Term set: ",words)
    print("MFs of term set: ",MFs)
    print("MF centroids of term set: ", CentroidsMFs)
    print("YLWA", YLWA)
    print("Similarity matrix: ", S)
    print("Indices of similarity terms: ", indices)
    print("Most similar linguistic term is: ", decode)"""
    print("Label of the complete face is: ", face_label)

