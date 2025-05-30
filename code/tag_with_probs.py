"""
This file contains functions that get probabilities of each tag (rather than just the most likely tag) for each sentence.
"""

import os
from openai import OpenAI
from prompts import chat_prompt_templates
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pyprojroot import here
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backofffrom pyprojroot import here


def format_messages(prompt, test_sentence):
    """
    Turn a LangChain prompt into a list of dictionaries to pass to openai.
    """
    messages = prompt.format_prompt(test_sentence=test_sentence).to_messages()
    message_dicts = [msg.dict() for msg in messages]
    for msg in message_dicts:
        del msg["additional_kwargs"]
        if msg["type"] == "ai":
            msg["type"] = "assistant"
        if msg["type"] == "human":
            msg["type"] = "user"
        msg["role"] = msg["type"]
        del msg["type"]
        if "example" in msg:
            del msg["example"]
    return message_dicts


openai = OpenAI()


tag_first_tokens = {
    "abstract": {"abstract": "abstract", "con": "concrete", "ign": "ignorance"},
    "policy": {"not": "not policy", "policy": "policy"},
    "dynamics": {"not": "not dynamics", "d": "dynamics"},
    "valence": {"win": "winning", "losing": "losing", "neutral": "neutral"},
}


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)


def get_tag_probabilities(test_sentence, prompt, tag_type, model="gpt-3.5-turbo"):
    """
    Get the probability of each tag for a given sentence.
    """
    response = completion_with_backoff(
        model=model,
        messages=format_messages(prompt, test_sentence),
        logprobs=True,
        top_logprobs=len(tag_first_tokens[tag_type]),
        n=1,
        max_tokens=1,
        temperature=0,
    )

    log_prob_dict = {}
    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    for log_prob in logprobs:
        if log_prob.token in tag_first_tokens[tag_type]:
            log_prob_dict[tag_first_tokens[tag_type][log_prob.token]] = log_prob.logprob

    # if any of the tags are missing, set their probability to 0
    for tag in tag_first_tokens[tag_type].values():
        if tag not in log_prob_dict:
            log_prob_dict[tag] = -np.inf

    return log_prob_dict


def tag_sentences_prob(sentences, model_name):
    rows = []
    for sentence in sentences:
        row = {"sentence": sentence}
        for tag_type in tag_first_tokens.keys():
            log_prob_dict = get_tag_probabilities(
                sentence, chat_prompt_templates[tag_type], tag_type, model_name
            )
            for val, log_prob in log_prob_dict.items():
                row[f"{tag_type}_{val}"] = log_prob
        rows.append(row)

    return pd.DataFrame(rows)


parser = ArgumentParser()

parser.add_argument(
    "--messages_path",
    type=str,
    default="data/raw-data/tagged_annotated_sentences.csv",
)
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--output_filename", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    prompt_types = chat_prompt_templates.keys()
    messages_path = here() / args.messages_path

    # Load in the messages
    df_messages = pd.read_csv(messages_path)
    sentences = df_messages["sentence"].drop_duplicates()

    n_batches = int(np.ceil(len(sentences) / args.batch_size))
    print(f"tagging {len(sentences)} sentences...")

    df_total = pd.DataFrame()
    for i in range(n_batches):
        min_sentence, max_sentence = i * args.batch_size, min(
            (i + 1) * args.batch_size, len(sentences)
        )
        sentence_batch = sentences[min_sentence:max_sentence]
        df_tagged = tag_sentences_prob(sentence_batch, args.model_name)
        df_tagged.to_csv(
            here(f"data/partial/tagged_sentences_{min_sentence}-{max_sentence}.csv"),
            index=False,
        )
        df_total = pd.concat([df_total, df_tagged])

    df_total.to_csv(here(f"data/{args.output_filename}.csv"), index=False)
