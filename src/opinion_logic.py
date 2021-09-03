# -*- coding: utf-8 -*-

import copy

# import os
import time

import pandas as pd
import numpy as np
import spacy
from textblob_de import TextBlobDE as TextBlob

from src.keywords import ANTI_NUCLEAR, CONSERVATIVE_ENERGY, NEUTRAL_ENERGY, PRO_NUCLEAR
from src.utils import extend_topics, parse_config

TOPICS = [ANTI_NUCLEAR, PRO_NUCLEAR, NEUTRAL_ENERGY, CONSERVATIVE_ENERGY]
TOPICS_EXTENDED = extend_topics(copy.deepcopy(TOPICS))
ALL_KEYWORDS_EXTENDED = [y for x in TOPICS_EXTENDED for y in x]

CONFIG = parse_config()


class Speech:
    """
    Holds a single speech with all accompanying information and provides
    methods to process the speech.
    """

    def __init__(self, doc_index, doc) -> None:
        self.doc_index = doc_index
        self.doc = doc
        self.speech_length = len(doc)
        self.keyword_counter = 0
        self.keywords = []
        self.negation = []
        self.third_person = []

        self.AN_descriptive = []
        self.AN_s = 0
        self.PN_descriptive = []
        self.PN_s = 0
        self.NE_descriptive = []
        self.NE_s = 0
        self.CE_descriptive = []
        self.CE_s = 0

        # COUNTERS
        self.subtree_length_counter = 0
        self.ancestors_length_counter = 0
        self.negation_counter = 0
        self.third_person_counter = 0
        self.score_neutral_counter = 0
        self.score_positive_counter = 0
        self.score_negative_counter = 0

        # CACHE ATTRIBUTES
        self.active_token = None
        self.active_topic = None
        self.active_negation = None
        self.active_third_person = None

    def as_dict(self):
        """
        Access important attributes as a dictionary

        Returns
        -------
        Dict
            Dictionary of most relevant speech attributes
        """
        return {
            "doc_index": self.doc_index,
            "speech_length": self.speech_length,
            "keyword_counter": self.keyword_counter,
            "keywords": self.keywords,
            "AN_descriptive": self.AN_descriptive,
            "AN_s": self.AN_s,
            "PN_descriptive": self.PN_descriptive,
            "PN_s": self.PN_s,
            "NE_descriptive": self.NE_descriptive,
            "NE_s": self.NE_s,
            "CE_descriptive": self.CE_descriptive,
            "CE_s": self.CE_s,
            "subtree_length_counter": self.subtree_length_counter,
            "ancestors_length_counter": self.ancestors_length_counter,
            "negation_counter": self.negation_counter,
            "third_person_counter": self.third_person_counter,
            "score_neutral_counter": self.score_neutral_counter,
            "score_positive_counter": self.score_positive_counter,
            "score_negative_counter": self.score_negative_counter,
        }

    def _negation(self, relatives):
        self.active_negation = []

        for i in range(0, len(relatives)):
            if "PTKNEG" in [child.tag_ for child in relatives[i].children]:
                self.active_negation.append(-1)
                self.negation_counter += 1
            else:
                self.active_negation.append(1)

    def _third_person(self, relatives):
        self.active_third_person = []

        for i in range(0, len(relatives)):
            is_ent = 0
            for token in relatives[i].ancestors:
                if token.dep_ == "ROOT":
                    children = [child for child in token.children]
                    for token in children:
                        if token.ent_iob == 3:
                            is_ent += 1

            if is_ent > 0:
                self.active_third_person.append(0)
                self.third_person_counter += 1
            else:
                self.active_third_person.append(1)

    def _score_relatives(self, relatives):
        sentiment_list = []
        for i in range(0, len(relatives)):
            # Extracts the lemma of word i in the list

            descriptor = TextBlob(relatives[i].lemma_)
            descriptor_sentiment = descriptor.sentiment.polarity

            #  Multiplies with the negation sign (either 1 or -1)
            descriptor_sentiment = descriptor_sentiment * self.active_negation[i]
            # Multiplies with the third person attribution value (0 or 1)
            descriptor_sentiment = descriptor_sentiment * self.active_third_person[i]

            # Tracks score taken
            if descriptor_sentiment == 0:
                self.score_neutral_counter += 1
            if descriptor_sentiment > 0:
                self.score_positive_counter += 1
            if descriptor_sentiment < 0:
                self.score_negative_counter += 1

            sentiment_list = []
            sentiment_list.append(relatives[i].lemma_)
            sentiment_list.append(descriptor_sentiment)
            sentiment_list.append(self.active_negation[i])
            sentiment_list.append(self.active_third_person[i])

            if self.active_topic == 0:
                self.AN_descriptive.append(sentiment_list)
                self.AN_s += descriptor_sentiment
            elif self.active_topic == 1:
                self.PN_descriptive.append(sentiment_list)
                self.PN_s += descriptor_sentiment
            elif self.active_topic == 2:
                self.NE_descriptive.append(sentiment_list)
                self.NE_s += descriptor_sentiment
            elif self.active_topic == 3:
                self.CE_descriptive.append(sentiment_list)
                self.CE_s += descriptor_sentiment

    def _subtree(self):
        # direct_subtree is a list of the class spacy.tokens.token.Token
        self.subtree = [
            descendant
            for descendant in self.active_token.subtree
            if descendant.tag_ not in ["PTKNEG"]
            and descendant.text not in ALL_KEYWORDS_EXTENDED
            and descendant.pos_ in CONFIG["relevant_scoring_pos"]
            # and descendant.tag_ in CONFIG["relevant_scoring_tags"]
        ]

    def _ancestors(self):
        self.ancestors = [
            ancestor
            for ancestor in self.active_token.ancestors
            if ancestor.tag_ not in ["PTKNEG"]
            and ancestor.pos_ in CONFIG["relevant_scoring_pos"]
            # and ancestor.tag_ in CONFIG["relevant_scoring_tags"]
        ]

    def _process_keyword_token(self):
        self.keyword_counter += 1
        self.keywords.append(self.active_token.text)

        # SUBTREE
        self._subtree()
        # On non-empty subtree lists, the score calculation function is called
        if self.subtree != []:
            self._negation(self.subtree)
            self._third_person(self.subtree)
            self.subtree_length_counter += len(self.subtree)
            self._score_relatives(self.subtree)

        # ANCESTORS
        self._ancestors()
        # On non-empty subtree lists, the score calculation function is called
        if self.ancestors != []:
            self._negation(self.ancestors)
            self._third_person(self.ancestors)
            self.ancestors_length_counter += len(self.ancestors)
            self._score_relatives(self.ancestors)

    def _process_doc(self):
        start_time = time.time()
        for token in self.doc:
            self.active_token = token
            for j in range(0, len(TOPICS_EXTENDED)):
                if token.lemma_ in TOPICS_EXTENDED[j]:
                    self.active_topic = j
                    self._process_keyword_token()
                    break  # no duplicate matches needed for a single token
            self.active_token = None
            self.active_topic = None
            self.active_negation = None
            self.active_third_person = None
        self.duration = time.time() - start_time


def scoring(df):
    """
    Applies opinion score logic.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results of parsed speech.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional score columns
    """
    # Sets all children score columns to equal the parents' values
    df["AN_sp"] = df["AN_s"]
    df["AN_sa"] = df["AN_s"]

    df["PN_sp"] = df["PN_s"]
    df["PN_sa"] = df["PN_s"]

    df["NE_sp"] = df["NE_s"]
    df["NE_sa"] = df["NE_s"]

    df["CE_sp"] = df["CE_s"]
    df["CE_sa"] = df["CE_s"]

    df["score"] = df["AN_s"] - df["PN_s"] - df["CE_s"]
    df["score_p"] = df["AN_sp"] - df["PN_sp"] - df["CE_sp"]
    df["score_a"] = df["AN_sa"] - df["PN_sa"] - df["CE_sa"]

    surge_duration = 400
    surge = 2 * -np.cos(1 / 400 * np.pi * np.arange(0, surge_duration)) + 3
    surge_start = df.query("electoral_term==17 and session==94").index[0] - (
        surge_duration / 2
    )
    surge_end = surge_start + surge_duration
    weights = np.concatenate(
        (np.ones(int(surge_start)), surge, np.ones(int(len(df) - surge_end)))
    )
    assert len(weights) == len(df), "Weights do not match df"

    df["weight"] = weights
    df["w_score"] = df["score"] / df["weight"]
    df["w_score_p"] = df["score_p"] / df["weight"]
    df["w_score_a"] = df["score_a"] / df["weight"]

    return df


def run_opinion_logic(df, subset_start=None, subset_end=None, compute_scores=True):
    """
    Executes the algorithm serially.

    The algorithm checks for the opinion towards energy politics and
    whether the spoken content is attributed to the speaker or a third
    person.

    Parameters
    ----------
    subset_start : int, optional
        Index value to start processing, by default None
    subset_end : int, optional
        Index value to finish processing, by default None

    Returns
    -------
    pd.DataFrame
        Results of the algorithm concatenated with the original speeches
        dataframe.
    """

    if subset_start is not None and subset_end is not None:
        df = df[subset_start:subset_end]
    df = df.reset_index()

    gerNLP = spacy.load(CONFIG["spacy_language_model"])
    docs = gerNLP.pipe(df["text"], disable=["ner", "attribute_ruler"])
    ix = df["index"]

    results = []
    for count, doc in enumerate(docs):

        speech = Speech(ix[count], doc)
        speech._process_doc()
        results.append(speech)

    df_result = pd.DataFrame([result.as_dict() for result in results])

    assert not df_result.empty, "No results computed!"

    df = pd.merge(df, df_result, how="right", left_on="index", right_on="doc_index")

    if compute_scores:
        print("Computing scores now")
        df = scoring(df)
        df.to_pickle(CONFIG["processed_df_cache"])
        print(
            f"Processed speeches incl. scores cached to {CONFIG['processed_df_cache']}"
        )
    else:
        print("Speeches processed, no scores calculated -> results not cached")

    return df


# -------------------------------< Main Start >-------------------------------#
# Main is for debugging purposes
if __name__ == "__main__":
    results = run_opinion_logic()
