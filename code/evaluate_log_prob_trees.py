"""
Fit and evaluate decision trees that classify sentences according to the log probability of each tag.
"""
import os
import numpy as np
import pandas as pd
from prompts import examples
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, KFold
from pyprojroot import here


def argmax_accuracy(X, y):
    """
    Compute accuracy of argmax classifier.
    """
    y_pred = np.argmax(X, axis=1)
    accuracy = accuracy_score(y_pred, y)
    f1_val = f1_score(y_pred, y, average="weighted")
    conf_mat = confusion_matrix(y_pred, y)
    return accuracy, f1_val, conf_mat


def fit_log_prob_tree(X, y):
    X = np.array(X)
    X = np.maximum(X, -99)
    clf = DecisionTreeClassifier(max_depth=X.shape[1] - 1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y_pred, y)
    conf_mat = confusion_matrix(y_pred, y)
    return clf, accuracy, conf_mat


def recode_human_labels(df):
    # re-encode human labels to match tags
    df["abstract_human"] = df["abstract_human"].apply(
        lambda x: "abstract"
        if x == "abstract, complex, high-level information"
        else "concrete"
        if x == "concrete, simple, low-level information"
        else "ignorance"
    )
    df["policy_human"] = df["policy_human"].apply(
        lambda x: "policy"
        if x == "policy, or what actions to take including strategies or instructions"
        else "not policy"
    )
    df["dynamics_human"] = df["dynamics_human"].apply(
        lambda x: "dynamics"
        if x == "dynamics, or how the world works including explanations or affordances"
        else "not dynamics"
    )
    df["valence_human"] = df["valence_human"].apply(
        lambda x: "winning"
        if x
        == "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies"
        else "neutral"
        if x == "neutral information"
        else "losing"
    )


TAGS_WITH_VALUES = {
    "abstract": ("abstract", "concrete", "ignorance"),
    "policy": ("policy", "not policy"),
    "dynamics": ("dynamics", "not dynamics"),
    "valence": ("winning", "losing", "neutral"),
}

if __name__ == "__main__":
    # load data
    df_tagged = pd.read_csv(here("data/tagged_sentences_prob.csv"))
    df_human = pd.read_csv(here("data/raw-data/tagged_annotated_sentences.csv"))
    df = pd.merge(df_tagged, df_human, on="sentence", suffixes=("_lm", "_human"))
    recode_human_labels(df)

    # fit and evaluate decision trees
    for tag_type, tag_values in TAGS_WITH_VALUES.items():
        print(tag_type.upper())
        print(f"tag values: {tag_values}")

        # filter out the sentences used in the few-shot examples
        df_tag = df[~df["sentence"].isin([x["sentence"] for x in examples[tag_type]])]

        # compute feature matrix and label vector
        X = np.array(df_tag[[f"{tag_type}_{value}" for value in tag_values]])
        X = np.maximum(X, -99)
        y = np.array([tag_values.index(tag) for tag in df_tag[f"{tag_type}_human"]])

        # get the accuracy of the argmax classifier
        accuracy, f1_val, conf_mat = argmax_accuracy(X, y)
        print(
            f"argmax classifier:\naccuracy: {100 * accuracy:.1f}%\nconfusion matrix:\n{conf_mat}"
        )

        # decision tree cross-validated accuracy, using LOO cross-validation
        clf = RandomForestClassifier(
            max_depth=X.shape[1], n_estimators=5, random_state=2024
        )

        y_pred = cross_val_predict(
            clf, X, y, cv=KFold(n_splits=y.shape[0], shuffle=True, random_state=2024)
        )
        accuracy = accuracy_score(y_pred, y)
        conf_mat = confusion_matrix(y_pred, y)
        f1_val = f1_score(y_pred, y, average="weighted")
        print(
            f"cross-validated random forest:\naccuracy: {100 * accuracy:.1f}%\nconfusion matrix:\n{conf_mat}"
        )

        print("\n")
