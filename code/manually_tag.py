"""
A little command-line script to manually tag sentences
"""
import os
import pandas as pd
from pyprojroot import here

all_tags = {
    "relevant": {
        "question": "Does the sentence contain specific advice relevant to the game?",
        "options": ["y", "n"],
        "option_names": ["true", "false"],
    },
    "dynpol": {
        "question": "Is the sentence about dynamics (how the world works, including explanations or affordances), policy (what the agent should do), or both?",
        "options": ["d", "p", "b"],
        "option_names": ["dynamics", "policy", "both"],
    },
    "abstract_nouns": {
        "question": "Does the sentence use noun-based abstractions, like 'teleporter' or 'walls'? Sentences without abstract nouns tend to talk exclusively about colored squares and blocks.",
        "options": ["y", "n"],
        "option_names": ["true", "false"],
    },
    "abstract_verbs": {
        "question": "Does the sentence use abstract or analogical verbs, like talking about 'eating' blocks? Sentences without verbal abstraction tend to talk exclusively about blocks moving or touching each other.",
        "options": ["y", "n"],
        "option_names": ["true", "false"],
    },
    "valence": {
        "question": "Does the sentence use mostly positive language, including talk of winning and goals, mostly negative language, including talk of dying and losing lives, or mostly neutral language?",
        "options": ["p", "n", "u"],
        "option_names": ["positive", "negative", "neutral"],
    },
    "uncertainty": {
        "question": 'Does the sentence contain expressions of uncertainty, including phrases like "I think" or "I don\'t know"?',
        "options": ["y", "n"],
        "option_names": ["true", "false"],
    },
}


def get_tag(tag_name, print_question=True):
    tag = all_tags[tag_name]
    question = f"{tag['question']} ({'/'.join(tag['options'])})"
    if print_question:
        print(question)
    answer = None
    while answer not in tag["options"]:
        answer = input()

    return tag_name + ":" + tag["option_names"][tag["options"].index(answer)]


def get_tags(sent):
    relevant = get_tag("relevant")
    tags = [relevant]
    valence = get_tag("valence")
    tags.append(valence)
    uncertainty = get_tag("uncertainty")
    tags.append(uncertainty)
    if relevant == "relevant:true":
        dynpol = get_tag("dynpol")
        tags.append(dynpol)
        abstract_nouns = get_tag("abstract_nouns")
        tags.append(abstract_nouns)
        abstract_verbs = get_tag("abstract_verbs")
        tags.append(abstract_verbs)

    return tags


if __name__ == "__main__":
    sentences = pd.read_csv(here("data/raw-data/all_tagged_sentences.csv"))

    print(f"How many sentences do you want to tag?")
    n_sentences = int(input())

    manually_tagged_rows = []
    i = 1
    if not os.path.isfile(here("data/raw-data/manually_tagged_sentences_partial.csv")):
        with open(
            here("data/raw-data/manually_tagged_sentences_partial.csv"), "w"
        ) as f:
            f.write("sentence,tags\n")
    for index, row in sentences.sample(n=n_sentences).iterrows():
        sent = row["sentence"]
        print(f"sentence {i} of {n_sentences}:\n\n{sent}\n")
        tags = get_tags(sent)
        manually_tagged_rows.append({"sentence": sent, "tags": ";".join(tags)})
        with open(
            here("data/raw-data/manually_tagged_sentences_partial.csv"), "a"
        ) as partial_rows:
            partial_rows.write(f"{sent},{';'.join(tags)}\n")
        i += 1

    df_tagged = pd.DataFrame(manually_tagged_rows)
    df_tagged.to_csv(here("data/raw-data/ben_manually_tagged_sentences.csv"))
