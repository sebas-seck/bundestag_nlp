# -*- coding: utf-8 -*-

import pandas as pd
import yaml
from IPython.display import display


def parse_config():
    """
    Parses the default config file.

    Returns
    -------
    Dict
        Configured keys and their values
    """
    config = "configs/default.yaml"
    with open(config) as f:
        return yaml.safe_load(f)


CONFIG = parse_config()


def show(table):
    """
    Displays table without hidden columns.

    Parameters
    ----------
    table : pd.DataFrame
        Any Pandas dataframe
    """
    with pd.option_context("display.max_columns", None):
        # with pd.option_context("display.max_colwidth", None, 'display.max_columns', None):
        display(table)


def extend_topics(topics):
    """
    Extends a lists' strings with the lemmatized values of the list.

    Parameters
    ----------
    topics : List
        List of topics without lemma values for each list item

    Returns
    -------
    List
        Extended topic list
    """
    # Extends list of list topic models by lemmatized values of existing content
    import spacy

    gerNLP = spacy.load(CONFIG["spacy_language_model"])

    for topic in topics:
        for i in range(0, len(topic)):
            doc = gerNLP(topic[i])  # creates the spaCy document per word
            if (
                doc != doc[0].lemma_
            ):  # states condition: lemma has to differ from existing item value
                topic.extend(
                    [doc[0].lemma_]
                )  # adds the lemmatized value to the respective topic model list
    return topics
