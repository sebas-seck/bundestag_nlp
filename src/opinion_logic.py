# -*- coding: utf-8 -*-

import time

import nltk
import spacy
from textblob_de import TextBlobDE as TextBlob

from src.keywords import ANTI_NUCLEAR, CONSERVATIVE_ENERGY, NEUTRAL_ENERGY, PRO_NUCLEAR

# nltk.download('punkt')

TOPICS = [ANTI_NUCLEAR, PRO_NUCLEAR, NEUTRAL_ENERGY, CONSERVATIVE_ENERGY]
ALL_KEYWORDS = ANTI_NUCLEAR + PRO_NUCLEAR + NEUTRAL_ENERGY + CONSERVATIVE_ENERGY


class OpinionAnalyzer(object):
    def __init__(self, df) -> None:
        self.df = df

        # Initialization of Scores
        # Anti Nuclear Descriptive & Score
        self.df["AN_descriptive"] = ""
        self.df["AN_descriptive"] = self.df["AN_descriptive"].apply(list)
        self.df["AN_s"] = 0
        self.df["AN_sp"] = 0
        self.df["AN_s_afer"] = 0

        # Pro Nuclear Descriptive & Score
        self.df["PN_descriptive"] = ""
        self.df["PN_descriptive"] = self.df["PN_descriptive"].apply(list)
        self.df["PN_s"] = 0
        self.df["PN_sp"] = 0
        self.df["PN_sa"] = 0

        # Neutral Energy Descriptive & Score
        self.df["NE_descriptive"] = ""
        self.df["NE_descriptive"] = self.df["NE_descriptive"].apply(list)
        self.df["NE_s"] = 0
        self.df["NE_sp"] = 0
        self.df["NE_sa"] = 0

        # Conservative Energy Descriptive & Score
        self.df["CE_descriptive"] = ""
        self.df["CE_descriptive"] = self.df["CE_descriptive"].apply(list)
        self.df["CE_s"] = 0
        self.df["CE_sp"] = 0
        self.df["CE_sa"] = 0

        # Placeholders for later final score calculation
        self.df["score"] = 0
        self.df["score_p"] = 0
        self.df["score_a"] = 0

        # Column to log scoring
        self.df["score_log"] = ""
        self.df["score_log"] = self.df["score_log"].apply(list)

        self.df["speech_length"] = None

        # counters
        self.keyword_counter = 0
        self.negation_counter = 0
        self.third_person_counter = 0
        self.score_neutral_counter = 0
        self.score_positive_counter = 0
        self.score_negative_counter = 0
        self.subtree_length_counter = 0
        self.ancestors_length_counter = 0

        self.sentiment_list = None
        self.protocol = None

    def calc_scores(self, subtree, negation, third_person):

        score = 0
        sentiment_list = []
        for i in range(0, len(subtree)):
            # Extracts the lemma of word i in the list
            descriptor = TextBlob(subtree[i].lemma_)
            # Looks up the sentiment polarity score in the dictionary
            descriptor_sentiment = descriptor.sentiment[0]

            # Tracks score taken
            if descriptor_sentiment == 0:
                self.score_neutral_counter += 1
            if descriptor_sentiment > 0:
                self.score_positive_counter += 1
            if descriptor_sentiment < 0:
                self.score_negative_counter += 1

            # Multiplies with the negation sign (either 1 or -1)
            descriptor_sentiment_after_negation = descriptor_sentiment * negation[i]
            # Multiplies with the third person attribution value (0 or 1)
            descriptor_sentiment_after_third_person = (
                descriptor_sentiment_after_negation * third_person[i]
            )
            # Adds the calculated value to the score
            score += descriptor_sentiment_after_negation

            # Creates the variable 'sentiment_list' for documentation of the procedure of each element of the subtree list
            self.sentiment_list.append(negation[i])
            self.sentiment_list.append(third_person[i])
            self.sentiment_list.append(subtree[i].lemma_)
            self.sentiment_list.append(descriptor_sentiment)

        return score

    def negation(self, list_of_words, negation):

        for i in range(0, len(list_of_words)):
            if "PTKNEG" in [child.tag_ for child in list_of_words[i].children]:
                negation.append(-1)
                self.negation_counter += 1
            else:
                negation.append(1)

        return negation

    def third_person(self, list_of_words, third_person):
        """Checks if the speaker intends to requote another person and returns a binary result"""

        for i in range(0, len(list_of_words)):
            is_ent = 0
            for token in list_of_words[i].ancestors:
                if token.dep_ == "ROOT":
                    children = [child for child in token.children]
                    for token in children:
                        if token.ent_iob == 3:
                            is_ent += 1

            if is_ent > 0:
                third_person.append(0)
                self.third_person_counter += 1
            else:
                third_person.append(1)

        return third_person

    def subtree(self, token):

        negation = []
        third_person = []

        # direct_subtree is a list of the class spacy.tokens.token.Token
        subtree = [
            descendant
            for descendant in token.subtree
            if descendant.tag_ not in ["PTKNEG"] and descendant.text not in ALL_KEYWORDS
        ]
        # second level of direct subtree
        if subtree != []:
            negation = self.negation(subtree, negation)
            third_person = self.third_person(subtree, third_person)
            self.subtree_length_counter += len(subtree)

        return subtree, negation, third_person

    def ancestors(self, token):
        """Takes a token as input and returns descriptive children and the associated negation"""

        negation = []
        third_person = []

        # direct_subtree is a list of the class spacy.tokens.token.Token
        ancestors = [
            ancestor for ancestor in token.ancestors if ancestor.tag_ not in ["PTKNEG"]
        ]
        # second level of direct subtree
        if ancestors != []:
            negation = self.negation(ancestors, negation)
            third_person = self.third_person(ancestors, third_person)
            self.ancestors_length_counter += len(ancestors)

        return ancestors, negation, third_person

    def _batch(self, i, doc):

        for token in doc:
            self.df.loc[i, "speech_length"] = len(doc)
            for j in range(0, len(TOPICS)):
                if token.lemma_ in TOPICS[j]:

                    self.keyword_counter += 1

                    self.sentiment_list = []

                    # SUBTREE
                    # Initializes the intermediate score for this tokens' subtree to 0
                    token_score_subtree = 0
                    # Calls the subtree function which returns a list of words and a list of negations
                    subtree, negation, third_person = self.subtree(token)
                    # On non-empty subtree lists, the score calculation function is called
                    if subtree != []:
                        token_score_subtree = self.calc_scores(
                            subtree, negation, third_person
                        )

                    # ANCESTORS
                    # Initializes intermediate score for this tokens' ancestors description to 0
                    token_score_ancestors = 0
                    # Calls the head description functions which returns a list of words and a list of negations
                    ancestors, negation, third_person = self.ancestors(token)
                    # On non-empty head_description lists, the score calculation function is called
                    if ancestors != []:
                        token_score_ancestors = self.calc_scores(
                            ancestors, negation, third_person
                        )

                    token_score = token_score_subtree + token_score_ancestors

                    if j == 0:
                        self.df.loc[i, "AN_descriptive"].append(self.sentiment_list)
                        self.df.loc[i, "AN_s"] += token_score
                    elif j == 1:
                        self.df.loc[i, "PN_descriptive"].append(self.sentiment_list)
                        self.df.loc[i, "PN_s"] += token_score
                    elif j == 2:
                        self.df.loc[i, "NE_descriptive"].append(self.sentiment_list)
                        self.df.loc[i, "NE_s"] += token_score
                    elif j == 3:
                        self.df.loc[i, "CE_descriptive"].append(self.sentiment_list)
                        self.df.loc[i, "CE_s"] += token_score

    def main(self):
        start_time = time.time()

        self.gerNLP = spacy.load("de_core_news_lg")

        for i, doc in enumerate(self.gerNLP.pipe(self.df["text"], batch_size=100)):

            self._batch(i, doc)
            if i == self.df.shape[0]:
                break

        end_time = time.time()

        duration = end_time - start_time
        self.protocol = (
            "The algorithm processing duration was "
            + str(round(duration / 60, 1))
            + " minutes."
        )
        self.protocol += (
            "\n" + str(self.keyword_counter) + " occurances of keywords were identified"
        )
        self.protocol += (
            "\n"
            + str(self.subtree_length_counter)
            + " words within keywords' subtrees were checked."
        )
        self.protocol += (
            "\n"
            + str(self.ancestors_length_counter)
            + " words within keywords' ancestors were checked."
        )
        self.protocol += (
            "\n"
            + str(self.third_person_counter)
            + " keywords were used in statements attributed to somebody else by the speaker."
        )
        self.protocol += (
            "\n" + str(self.negation_counter) + " keywords descriptions were negated."
        )
        self.protocol += (
            "\n"
            + str(self.score_neutral_counter)
            + " descriptions of keywords were neutral."
        )
        self.protocol += (
            "\n"
            + str(self.score_negative_counter)
            + " descriptions of keywords were negative."
        )
        self.protocol += (
            "\n"
            + str(self.score_positive_counter)
            + " descriptions of keywords were positive."
        )

        return self.df
