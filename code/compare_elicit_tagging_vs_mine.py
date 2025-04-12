"""
This file compares the original Elicit-based tagging schema against my attempt to re-implement it.
"""

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pyprojroot import here
from sklearn.metrics import confusion_matrix

parser = ArgumentParser()
parser.add_argument(
    "--elicit_filepath",
    type=str,
    default="data/raw-data/unique_sentences_tagged_9-25.csv",
)
parser.add_argument(
    "--retag_filepath", type=str, default="data/retagged_elicit_sentences.csv"
)

if __name__ == "__main__":
    args = parser.parse_args()
    df_elicit = pd.read_csv(here(args.elicit_filepath))
    df_retag = pd.read_csv(here(args.retag_filepath))
    print(df_elicit.columns)
    print(df_retag.columns)

    df_merged = df_elicit.merge(
        df_retag, on="sentence", how="inner", suffixes=(None, "_retag")
    )
    print(df_merged.columns)

    for feature in ("dynamics", "policy", "valence", "abstract"):
        retag_labels = df_merged[f"{feature}"]
        elicit_labels = df_merged[f"{feature}_elicit_label"]
        accuraccy = sum(retag_labels == elicit_labels) / len(retag_labels)

        conf_mat = confusion_matrix(elicit_labels, retag_labels)
        print(feature.upper())
        print(f"overall accuracy: {accuraccy*100}%")
        print(conf_mat)
