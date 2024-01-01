"""
This file makes confusion matrices for the human-tagged variables
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from pyprojroot import here

if __name__ == "__main__":
    df_human = pd.read_csv(here("data/raw-data/unique_sentences_tagged_9-25.csv"))
    print(df_human.columns)
    df_human = (
        df_human.filter(regex=r"elicit_label|human_label|sentence")
        .drop("sentence_number", axis=1)
        .drop_duplicates()
    )
    df_human = df_human[~df_human["dynamics_human_label"].isna()]
    print(f"n human-tagged sentences: {len(df_human)}")
    # df_human.columns = df_human.columns.str.removesuffix("_human_label")
    print(df_human.columns)

    df_lm = pd.read_csv(here("data/raw-data/tagged_annotated_sentences.csv"))
    df_merged = pd.merge(df_human, df_lm, on="sentence")

    for category in ("valence", "abstract", "dynamics", "policy"):
        cat_labels = tuple(df_merged[category].drop_duplicates())
        mine_vs_human_accuracy = accuracy_score(
            df_merged[f"{category}_human_label"], df_merged[category]
        )
        mine_vs_human_conf_mat = confusion_matrix(
            df_merged[f"{category}_human_label"], df_merged[category], labels=cat_labels
        )
        elicit_vs_human_accuracy = accuracy_score(
            df_merged[f"{category}_human_label"], df_merged[f"{category}_elicit_label"]
        )
        elicit_vs_human_conf_mat = confusion_matrix(
            df_merged[f"{category}_human_label"],
            df_merged[f"{category}_elicit_label"],
            labels=cat_labels,
        )
        mine_vs_elicit_accuracy = accuracy_score(
            df_merged[f"{category}_elicit_label"], df_merged[category]
        )
        mine_vs_elicit_conf_mat = confusion_matrix(
            df_merged[f"{category}_elicit_label"],
            df_merged[category],
            labels=cat_labels,
        )

        print(category.upper())
        print(cat_labels)
        print(f"My LM tagging vs human tagging:")
        print(f"accuracy: {mine_vs_human_accuracy*100}%")
        print(mine_vs_human_conf_mat)
        print(f"Elicit tagging vs human tagging:")
        print(f"accuracy: {elicit_vs_human_accuracy*100}%")
        print(elicit_vs_human_conf_mat)
        print(f"My LM tagging vs elicit tagging")
        print(f"accuracy: {mine_vs_elicit_accuracy*100}%")
        print(mine_vs_elicit_conf_mat)
