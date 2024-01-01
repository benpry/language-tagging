"""
Re-tag the sentences that were tagged through the Eliicit API, to see how closely our tagging system aligns.
"""
import os
from tagging_utils import sync_tag_all
from prompts import chat_prompt_templates
from pyprojroot import here
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()

parser.add_argument(
    "--messages_path",
    type=str,
    default="data/raw-data/unique_sentences_tagged_9-25.csv",
)
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")

if __name__ == "__main__":
    args = parser.parse_args()
    prompt_types = chat_prompt_templates.keys()

    messages_path = here() / args.messages_path

    # Load in the messages
    df_messages = pd.read_csv(messages_path)

    sentences = df_messages["sentence"].drop_duplicates()

    print(f"tagging {len(sentences)} sentences...")

    # split the sentences into tenths
    n = len(sentences) // 10

    # tag each subset of n // 10 sentences
    for i in range(10):
        if os.path.exists(here(f"data/partial/tags_subset_{i}.csv")):
            continue
        print(f"Tagging subset {i} of sentences")
        sentence_subset = sentences[i * n : (i + 1) * n]
        df_tagged = sync_tag_all(args.model_name, prompt_types, sentence_subset)
        df_tagged.to_csv(here(f"data/partial/tags_subset_{i}.csv"), index=False)

    # combine all the tagged sentences
    all_dfs_tagged = []
    for i in range(10):
        df_tagged = pd.read_csv(here(f"data/partial/tags_subset_{i}.csv"))
        all_dfs_tagged.append(df_tagged)

    # save the tagged sentences to csv
    df_tagged = pd.concat(all_dfs_tagged)
    df_tagged.to_csv(here(f"data/retagged_elicit_sentences.csv"), index=False)
