"""
Use log probs to classify the sentences
"""
import numpy as np
import pandas as pd
from prompts import examples
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from evaluate_log_prob_trees import recode_human_labels
from argparse import ArgumentParser
from pyprojroot import here


def fit_classifier(df_train, tag_type, tag_values):
    # filter out the sentences used in the few-shot examples
    df_tag = df_train[
        ~df_train["sentence"].isin([x["sentence"] for x in examples[tag_type]])
    ]

    # compute feature matrix and label vector
    X = np.array(df_tag[[f"{tag_type}_{value}" for value in tag_values]])
    X = np.maximum(X, -99)
    y = np.array([tag_values.index(tag) for tag in df_tag[f"{tag_type}_human"]])

    clf = RandomForestClassifier(
        max_depth=X.shape[1], n_estimators=5, random_state=2024
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    conf_mat = confusion_matrix(y_pred, y)

    return clf


def make_classifications(df_test, clf, tag_type, tag_values):
    X = np.array(df_test[[f"{tag_type}_{value}" for value in tag_values]])
    X = np.maximum(X, -99)

    y_pred = clf.predict(X)
    y_pred = [tag_values[i] for i in y_pred]
    return y_pred


def recode_tag_labels(df):
    # re-encode human labels to match tags
    df["abstract_tag"] = df["abstract_tag"].apply(
        lambda x: "abstract, complex, high-level information"
        if x == "abstract"
        else "concrete, simple, low-level information"
        if x == "concrete"
        else "ignorance statements or specific experiences"
    )
    df["policy_tag"] = df["policy_tag"].apply(
        lambda x: "policy, or what actions to take including strategies or instructions"
        if x == "policy"
        else "not policy, or what actions to take including strategies or instructions"
    )
    df["dynamics_tag"] = df["dynamics_tag"].apply(
        lambda x: "dynamics, or how the world works including explanations or affordances"
        if x == "dynamics"
        else "not dynamics, or how the world works including explanations or affordances"
    )
    df["valence_tag"] = df["valence_tag"].apply(
        lambda x: "winning, including mentions of scoring points, victory, success, goals, solutions, best strategies"
        if x == "winning"
        else "neutral information"
        if x == "neutral"
        else "losing, including information about death, losing points, lowering scores, forfeiting, losing lives, getting stuck or trapped"
    )


TAGS_WITH_VALUES = {
    "abstract": ("abstract", "concrete", "ignorance"),
    "policy": ("policy", "not policy"),
    "dynamics": ("dynamics", "not dynamics"),
    "valence": ("winning", "losing", "neutral"),
}


if __name__ == "__main__":
    df_tagged_human = pd.read_csv(here("data/tagged_sentences_prob_human.csv"))
    df_human = pd.read_csv(here("data/raw-data/tagged_annotated_sentences.csv"))
    df_train = pd.merge(
        df_tagged_human, df_human, on="sentence", suffixes=("_lm", "_human")
    )
    df_train = df_train.rename(
        columns={
            "dynamics": "dynamics_human",
            "policy": "policy_human",
            "abstract": "abstract_human",
            "valence": "valence_human",
        }
    )
    recode_human_labels(df_train)

    df_to_tag = pd.read_csv(here("data/tagged_sentences_prob.csv"))

    print(f"number of training examples: {len(df_train)}")
    print(f"number of test examples: {len(df_to_tag)}")

    # fit and evaluate decision trees
    for tag_type, tag_values in TAGS_WITH_VALUES.items():
        clf = fit_classifier(df_train, tag_type, tag_values)

        df_to_tag[f"{tag_type}_tag"] = make_classifications(
            df_to_tag, clf, tag_type, tag_values
        )

    recode_tag_labels(df_to_tag)
    df_to_tag.to_csv(
        here("data/tagged_sentences_prob_with_classifications.csv"), index=False
    )
