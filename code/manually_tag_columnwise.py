"""#!/usr/bin/env python
manually tag a bunch of sentences for one attribute
"""
import pandas as pd
from manually_tag import get_tag
from pyprojroot import here

category = "abstract_verbs"
if __name__ == "__main__":
    sentences = pd.read_csv(here("data/raw-data/all_tagged_sentences.csv"))["sentence"]

    for sentence in sentences[:10]:
        print(sentence)
        response = get_tag(category, print_question=False)
