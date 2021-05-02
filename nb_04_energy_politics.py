# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # External Shock on Energy Politics

# %%
import numpy as np
import pandas as pd
import time
import spacy
from math import log

# %%
gerNLP = spacy.load("de_core_news_lg")

# %%
df = pd.read_pickle("data/plpr.pkl")

# %%
# TOPIC: ANTI NUCLEAR
# Is <keyword> helping to leave nuclear energy (in a sustainable manner)? If yes, the topic fits here.
anti_nuclear = [
    "Atomausstieg",
    "Atomausstieges",
    "EEG",
    "EEG-Reform",
    "EEG-Umlage",
    "Energieleitungsausbaugesetz",
    "Energierevolution",
    "Energiespeicherung",
    "Energiewende",
    "Erdgasförderung",
    "Erneuerbare-Energien-Gesetz",
    "Erneuerbare-Energien-Gesetzes",
    "Gaskraftwerk",
    "Gaskraftwerke",
    "Gaskraftwerkes",
    "Gaskraftwerken",
    "Kernenergiegegner",
    "Kraft-Wärme-Kopplung",
    "Kraft-Wärme-Kopplungsgesetz",
    "KWK",
    "KWK-Gesetz",
    "KWKG",
    "Netzstabilität",
    "Onshorewindenergie",
    "Solaranlage",
    "Solaranlagen",
    "Solarenergie",
    "Solarstrom",
    "Solarstromes",
    "Solarzelle",
    "Solarzellen",
    "Sonnenenergie",
    "Speichermöglichkeit",
    "Speichermöglichkeiten",
    "Speichertechnologie",
    "Speichertechnologien",
    "Trassenausbau",
    "Wasserkraft",
    "Windenergie",
    "Ökostrom",
]

# %%
# TOPIC: PRO NUCLEAR
# Is <keyword> helping to keep nuclear energy? If yes, the topic fits here.
pro_nuclear = [
    "AKW",
    "AKWs",
    "Atomanlage",
    "Atomanlagen",
    "Atomaufsicht",
    "Atomenergie",
    "Atomenergiebehörde",
    "Atomforschung",
    "Atomgesetz",
    "Atomgesetzes",
    "Atomindustrie",
    "Atomkraftwerk",
    "Atomkraftwerke",
    "Atomkraftwerkes",
    "Atomlobby",
    "Atommüll",
    "Atomreaktor",
    "Atomreaktoren",
    "Atomreaktores",
    "Atomsicherheit",
    "Atomwirtschaft",
    "Brennelement",
    "Brennstäbe",
    "Endlager",
    "Endlagerung",
    "Euratom",
    "Euratom-Vertrag",
    "Kernenergie",
    "Kernenergielobby",
    "Kernenergiewirtschaft" "Kernkraft",
    "Kernkraftwerk",
    "Kernkraftwerke",
    "Kernkraftwerkes",
    "Kernreaktor",
    "Kernreaktoren",
    "Kernreaktors",
    "Laufzeitverlängerung",
    "Nuklearenergiekommission",
    "Nuklearmaterial",
    "Reaktor",
    "Reaktoren",
    "Reaktors",
    "Reaktorsicherheit",
    "Zwischenlagerung",
]

# %%
# TOPIC: CONSERVATIVE ENERGY
# Is <keyword> associated with conservative energy and does not fit into the above categories?
conservative_energy = [
    "Braunkohlekraftwerk",
    "Braunkohlekraftwerke",
    "Braunkohlestrom",
    "Braunkohleverstromung",
    "Einspeisevergütung",
    "Energiepreis",
    "Energiepreise",
    "Energiesparen",
    "Energiesparpotenzial",
    "Energiestrategie",
    "Kohle",
    "Kohlen",
    "Kohleausstieg" "Kohlekraftwerk",
    "Kohlekraftwerke",
    "Kohlekraftwerks",
    "Kohlelobby",
    "Netzbetreiber",
    "Primärenergie",
    "Steinkohlebergbau",
]

# %%
# TOPIC: NEUTRAL ENERGY
# Is <keyword> directly associated with energy politics but does not fit into the above categories?
neutral_energy = [
    "Brückentechnologie",
    "Brückentechnologien",
    "Emissionshandel",
    "Energie",
    "Energieeffizienz",
    "Energiemix",
    "Energiemixes",
    "Energiepolitik",
    "Energieprogramm",
    "Energiequelle",
    "Energiequellen",
    "Energiesteuer",
    "Energietechnologie",
    "Energietechnologien",
    "Energieunternehmen",
    "Energieversorgungsunternehmen",
    "Energiewirtschaft",
    "Energiewirtschaftsgesetz",
    "Erdverkabelung",
    "Kraftwerkbetreiber",
    "Kraftwerksbetreiber",
    "Netzentgelt",
    "Netzentgelte",
    "Strom",
    "Stromerzeugung",
    "Stromproduktion",
    "Versorgungssicherheit",
    "Überbrückungstechnologie",
    "Überbrückungstechnologien",
    "Übergangszeit",
]

# %%
import stylecloud

text = " ".join(anti_nuclear + pro_nuclear + neutral_energy + conservative_energy)
stylecloud.gen_stylecloud(
    text=text,
    icon_name="fas fa-atom",
    palette="colorbrewer.qualitative.Dark2_8",
    background_color="black",
    gradient="horizontal",
    output_name="docs/atom.png",
)

# %% [markdown]
# ![atom wordcloud](docs/atom.png)

# %%
# Organizes all topic models in a list of lists
topics = [anti_nuclear, pro_nuclear, neutral_energy, conservative_energy]


# %%
def topics_extension(topic):
    """Extends topic models by lemmatized values of existing content"""

    for i in range(0, len(topic)):
        doc = gerNLP(topic[i])  # creates the spaCy document per word
        if (
            doc != doc[0].lemma_
        ):  # states condition: lemma has to differ from existing item value
            topic.extend(
                [doc[0].lemma_]
            )  # adds the lemmatized value to the respective topic model list


# %%
# This cell calls the function just defined on all topic models organized through the list of lists "topics"
for i in range(0, len(topics)):
    topics_extension(topics[i])

# %%
# This cell organizes all keywords in a common list
all_keywords = []
all_keywords.extend(anti_nuclear)
all_keywords.extend(pro_nuclear)
all_keywords.extend(neutral_energy)
all_keywords.extend(conservative_energy)

# %%
df.shape

# %%
# If executable, this filters all speeches and keeps only those which include at least one word from the hardcoded keywords later in the notebook
df = df[df["text"].str.contains("|".join(all_keywords))]
df = df.reset_index(drop=True)

# %%
df.shape

# %% [markdown]
# ## Algorithm Preparation
# **Dummy Variable Pre External Shock / After External Shock**<br>
# The column *after_shock* is a dummy variable to indicate whether a speech fragment is part of a plenary meeting before the catastrophy in Fukushima or thereafter. The inflection point is between the plenary meetings 97 and 98 during the 17th legislative period. Refer to \ref{pol-dim-ger} \nameref{pol-dim-ger} for the conclusion how to choose the inflection point.

# %%
df["after_shock"] = np.where((df["sitzung"] <= 97) & (df["wahlperiode"] <= 17), 0, 1)

# %% [markdown]
# ## Named Entities Extraction
# The following chunk prepares the dataframe by adding three columns with empty lists to be populated with information about named entities.

# %%
df["NER"] = ""
df["NER"] = df["NER"].apply(list)
df["NER_text"] = ""
df["NER_text"] = df["NER_text"].apply(list)


# %%
def get_NER(i, doc):
    """Extracts named entities per speech fragment and stores them in a separate column"""

    for ent in doc.ents:
        df.loc[i, "NER"].append(ent.text + ";" + ent.label_)
    df.loc[i, "NER_text"] = len(df.loc[i, "NER"])


# %% [markdown]
# ### Opinion Analysis Algorithm
# #### Dataframe Preparation
# First things first, additional columns to store the returned information are needed. For each of the three categories, a column to store words and their sentiments are initialized, as well as a numeric column for the score.

# %%
def column_initialize_fun():
    """Initializes all columns required for scores & score calculations"""

    # Anti Nuclear Descriptive & Score
    df["AN_descriptive"] = ""
    df["AN_descriptive"] = df["AN_descriptive"].apply(list)
    df["AN_s"] = 0
    df["AN_sp"] = 0
    df["AN_s_afer"] = 0

    # Pro Nuclear Descriptive & Score
    df["PN_descriptive"] = ""
    df["PN_descriptive"] = df["PN_descriptive"].apply(list)
    df["PN_s"] = 0
    df["PN_sp"] = 0
    df["PN_sa"] = 0

    # Neutral Energy Descriptive & Score
    df["NE_descriptive"] = ""
    df["NE_descriptive"] = df["NE_descriptive"].apply(list)
    df["NE_s"] = 0
    df["NE_sp"] = 0
    df["NE_sa"] = 0

    # Conservative Energy Descriptive & Score
    df["CE_descriptive"] = ""
    df["CE_descriptive"] = df["CE_descriptive"].apply(list)
    df["CE_s"] = 0
    df["CE_sp"] = 0
    df["CE_sa"] = 0

    # Placeholders for later final score calculation
    df["score"] = 0
    df["score_p"] = 0
    df["score_a"] = 0

    # Column to log scoring
    df["score_log"] = ""
    df["score_log"] = df["score_log"].apply(list)


# %% [markdown]
# #### Occasion Counters
# In order to track occurances of cases, counters of ocassions are initialized.

# %%
def counter_initialize_fun():
    """Initializes counters to track occurances of steps within the algorithm"""

    global keyword_counter
    keyword_counter = 0

    global negation_counter
    negation_counter = 0

    global third_person_counter
    third_person_counter = 0

    global score_neutral_counter
    score_neutral_counter = 0

    global score_positive_counter
    score_positive_counter = 0

    global score_negative_counter
    score_negative_counter = 0

    global subtree_length_counter
    subtree_length_counter = 0

    global ancestors_length_counter
    ancestors_length_counter = 0


# %% [markdown]
# #### Sentiment Scoring with TextBlob

# %%
# imports the package TextBlob for sentiment analysis on German!
from textblob_de import TextBlobDE as TextBlob


def calc_scores(subtree, negation, third_person):
    """Takes a list of words and a list of associated negations and returns a total score of the list and a list documenting the score calculation"""

    global score
    score = 0
    for i in range(0, len(subtree)):
        # Extracts the lemma of word i in the list
        descriptor = TextBlob(subtree[i].lemma_)
        # Looks up the sentiment polarity score in the dictionary
        descriptor_sentiment = descriptor.sentiment[0]

        # Tracks score taken
        if descriptor_sentiment == 0:
            global score_neutral_counter
            score_neutral_counter += 1
        if descriptor_sentiment > 0:
            global score_positive_counter
            score_positive_counter += 1
        if descriptor_sentiment < 0:
            global score_negative_counter
            score_negative_counter += 1

        # Multiplies with the negation sign (either 1 or -1)
        descriptor_sentiment_after_negation = descriptor_sentiment * negation[i]
        # Multiplies with the third person attribution value (0 or 1)
        descriptor_sentiment_after_third_person = (
            descriptor_sentiment_after_negation * third_person[i]
        )
        # Adds the calculated value to the score
        score += descriptor_sentiment_after_negation

        # Creates the variable 'sentiment_list' for documentation of the procedure of each element of the subtree list
        sentiment_list.append(negation[i])
        sentiment_list.append(third_person[i])
        sentiment_list.append(subtree[i].lemma_)
        sentiment_list.append(descriptor_sentiment)

    # Returns two lists
    return sentiment_list
    return score


# %% [markdown]
# #### Negation Check
# This function checks for every token associated to a keyword, if it is being negated.

# %%
def negation_fun(list_of_words):
    """
    function takes list of keyword descriptors,
    checks if they are negated, returns list of same length
    with '1' indicating no negation and '-1' a negation of the keyword descriptor
    """

    for i in range(0, len(list_of_words)):
        if "PTKNEG" in [child.tag_ for child in list_of_words[i].children]:
            negation.append(-1)
            global negation_counter
            negation_counter += 1
        else:
            negation.append(1)

    return negation


# %% [markdown]
# #### Third-person Check
# This function checks whether the speaker attributes a sentence with a keyword to another person or organization.

# %%
def third_person_fun(list_of_words):
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
            global third_person_counter
            third_person_counter += 1
        else:
            third_person.append(1)

    return third_person


# %% [markdown]
# #### Retrieval of keyword descriptors
# The retrieval of words describing the keyword is the oxygen to the algorithm. Both quality and quantity impact the explanatory power of the algorithms' output.
# #### Descendants Function

# %%
def subtree_fun(token):
    """Takes a token as input and returns descriptive children and the associated negation"""

    global subtree
    global negation
    negation = []
    global third_person
    third_person = []

    # direct_subtree is a list of the class spacy.tokens.token.Token
    subtree = [
        descendant
        for descendant in token.subtree
        if descendant.tag_ not in ["PTKNEG"] and descendant.text not in all_keywords
    ]
    # second level of direct subtree
    if subtree != []:
        negation_fun(subtree)
        third_person_fun(subtree)
        global subtree_length_counter
        subtree_length_counter += len(subtree)

    return subtree
    return negation
    return third_person


# %% [markdown]
# #### Ancestors Function

# %%
def ancestors_fun(token):
    """Takes a token as input and returns descriptive children and the associated negation"""

    global ancestors
    global negation
    negation = []
    global third_person
    third_person = []

    # direct_subtree is a list of the class spacy.tokens.token.Token
    ancestors = [
        ancestor for ancestor in token.ancestors if ancestor.tag_ not in ["PTKNEG"]
    ]
    # second level of direct subtree
    if ancestors != []:
        negation_fun(ancestors)
        third_person_fun(ancestors)
        global ancestors_length_counter
        ancestors_length_counter += len(ancestors)

    return ancestors
    return negation
    return third_person


# %% [markdown]
# #### Descriptors Connected through Auxiliaries

# %%
def aux_connected_fun(token):
    """Bla bla bla"""

    global aux_connected
    global negation
    negation = []
    global third_person
    third_person = []

    if token.head.pos_ == "AUX":
        aux_connected = [
            child
            for child in token.head.children
            if child.text not in all_keywords and child.tag_ not in "PTKNEG"
        ]

    if aux_connected != []:
        negation_fun(aux_connected)
        third_person_fun(aux_connected)
        global aux_connected_length_counter
        aux_connected_length_counter += len(aux_connected)


# %% [markdown]
# #### Opinion Extraction

# %%
def get_opinion(i, doc):
    """Analyses a document for stated opinions"""

    for token in doc:
        for j in range(0, len(topics)):
            if token.lemma_ in topics[j]:

                global keyword_counter
                keyword_counter += 1

                global sentiment_list
                sentiment_list = []

                # SUBTREE
                # Initializes the intermediate score for this tokens' subtree to 0
                token_score_subtree = 0
                # Calls the subtree function which returns a list of words and a list of negations
                subtree_fun(token)
                # On non-empty subtree lists, the score calculation function is called
                if subtree != []:
                    calc_scores(subtree, negation, third_person)
                    token_score_subtree = score

                # ANCESTORS
                # Initializes intermediate score for this tokens' ancestors description to 0
                token_score_ancestors = 0
                # Calls the head description functions which returns a list of words and a list of negations
                ancestors_fun(token)
                # On non-empty head_description lists, the score calculation function is called
                if ancestors != []:
                    calc_scores(ancestors, negation, third_person)
                    token_score_ancestors = score

                #               # AUX CONNECTED
                #               # Initializes intermediate score for this tokens' auxiliary-connected description to 0
                #               token_score_aux_connected = 0
                #
                #               if token.head.pos_ == 'AUX':
                #                   aux_connected_fun(token)
                #                   if aux_connected != []:
                #                       calc_scores(aux_connected, negation, third_person)
                #                       token_score_aux_connected = score

                token_score = (
                    token_score_subtree + token_score_ancestors
                )  # + token_score_aux_connected

                if j == 0:
                    df.loc[i, "AN_descriptive"].append(sentiment_list)
                    df.loc[i, "AN_s"] += token_score
                elif j == 1:
                    df.loc[i, "PN_descriptive"].append(sentiment_list)
                    df.loc[i, "PN_s"] += token_score
                elif j == 2:
                    df.loc[i, "NE_descriptive"].append(sentiment_list)
                    df.loc[i, "NE_s"] += token_score
                elif j == 3:
                    df.loc[i, "CE_descriptive"].append(sentiment_list)
                    df.loc[i, "CE_s"] += token_score


# %%
def protocol(start_time, end_time):
    """Creates a protocol with metrics on the execution of the algorithm"""

    duration = end_time - start_time
    text = (
        "The algorithm processing duration was "
        + str(round(duration / 60, 1))
        + " minutes."
    )
    text += "\n" + str(keyword_counter) + " occurances of keywords were identified"
    text += (
        "\n"
        + str(subtree_length_counter)
        + " words within keywords' subtrees were checked."
    )
    text += (
        "\n"
        + str(ancestors_length_counter)
        + " words within keywords' ancestors were checked."
    )
    #   text += '\n'+str(aux_connected_length_counter) +' words connecting keywords and descriptors through auxiliaries were checked.'
    text += (
        "\n"
        + str(third_person_counter)
        + " keywords were used in statements attributed to somebody else by the speaker."
    )
    text += "\n" + str(negation_counter) + " keywords descriptions were negated."
    text += (
        "\n" + str(score_neutral_counter) + " descriptions of keywords were neutral."
    )
    text += (
        "\n" + str(score_negative_counter) + " descriptions of keywords were negative."
    )
    text += (
        "\n" + str(score_positive_counter) + " descriptions of keywords were positive."
    )
    print(text)


# %% [markdown]
# ## Algorithm Execution

# %%
import nltk

nltk.download("punkt")

# %%
# %%time

start_time = time.time()

# Initializes columns to store algorithm output
column_initialize_fun()

# Initializes counters to track occurances of steps within the algorithm
counter_initialize_fun()

for i, doc in enumerate(gerNLP.pipe(df["text"], batch_size=100)):

    get_NER(i, doc)
    get_opinion(i, doc)
    if i == df.shape[0]:
        break

end_time = time.time()

# %%
protocol(start_time, end_time)

# %% [markdown]
# ### Score Splitting at Inflection Point
# As it is essential to compare opinions before and after the external shock, both values extracted from the calculated scores in the various categories in conjunction with the dummy variable *after_shock*, which is 0 for pre-shock and 1 for after-shock speech fragments. <br>
# To do so, pre-shock and after-shock columns for each category are created (i.e. *NE_sp*, *PN_sa*). These are children to the main category score columns (i.e. *NE_s*, *PN_s*). Initially, the children columns take on the same value as the parents. Then, the values are adjusted to 0 if the *after_shock* variable value (0 or 1) does not match with the children score column suffices (pre or after). Accordingly, for all rows with a *after_shock* value of 1, the pre score will be reset to 0.

# %%
# Sets all children score columns to equal the parents' values
df["AN_sp"] = df["AN_s"]
df["AN_sa"] = df["AN_s"]

df["PN_sp"] = df["PN_s"]
df["PN_sa"] = df["PN_s"]

df["NE_sp"] = df["NE_s"]
df["NE_sa"] = df["NE_s"]

df["CE_sp"] = df["CE_s"]
df["CE_sa"] = df["CE_s"]

# %%
# Adjusts children score columns for their after_shock values
df.loc[df.after_shock == 1, ["AN_sp", "PN_sp", "NE_sp", "CE_sp"]] = 0
df.loc[df.after_shock == 0, ["AN_sa", "PN_sa", "NE_sa", "CE_sa"]] = 0

# %% [markdown]
# ### Score Calculation
# The main score is calculated using three of the four identified models in \ref{hardcoded-topic-modeling} \nameref{hardcoded-topic-modeling}. The score is designed to be positive to reflect progressiveness. Therefore, anti-nuclear energy opinions are added, pro-nuclear energy opinions are substracted, and conservative energy politics opinions are substracted as well, as those do not reflect the turnaround performed by politics. Solely opinions about energy politics which do not fall into any of the other three categories are not included in the score but are kept for reference.

# %%
df["score"] = df["AN_s"] - df["PN_s"] - df["CE_s"]
df["score_p"] = df["AN_sp"] - df["PN_sp"] - df["CE_sp"]
df["score_a"] = df["AN_sa"] - df["PN_sa"] - df["CE_sa"]

# %% [markdown]
# ### Delay Weight
# #### Weight Calculation
# The delay weight assigns a value between $1$ and $\log_{log\_base}(\infty)$ to each row. Later, the score will be divided by the weight to calculate the tenacity. The later a speech was given after the external shock, the less impact its score has. The delay weight will be useful in the calculation of the measure of tenacity.
# <br>
# The column *delay* is supposed to indicate how many sessions after the external shock a speech has been given. Values for speeches given before the external shock will be overwritten in the end as they are negative and will all be set to the same value.

# %%
# initializes the delay column to be similar to the running session number
df["delay"] = df["sitzung"]
# the auxiliary column stores an adjustment value which will be added to delay
# for speeches given in legislative period 17, 94 is substracted as the external shock took place inbetween the plenary meetings 94 and 95
df.loc[df.wahlperiode == 17, "aux_delay"] = -94
# for speeches given in legislative period 18, 159 is added as the external shock took place 159 plenary meetings before the first meeting in legislative period 18
df.loc[df.wahlperiode == 18, "aux_delay"] = 253 - 94
# the new variable reflects the adjustments from the chunk above
df["aux_delay_weight"] = df["delay"] + df["aux_delay"]
# sets the variable log_base
log_base = 3

# %% [markdown]
# The *log_base* variable is an important argument to the delay weight. Instead of using the delay in number of sessions to punish late speeches / opinion changes, I am choosing to apply a non-linear scale which does not completely invalidate late speeches. Further, it even promotes (attaches more weight to a speech compared to speeches given before the external shock) early speeches just after the external shock up to the point where $aux\_delay\_weight = log\_base$. Thus, choosing $log\_base = 3$ would
#
# As I do not want to punish any speech given before the external shock, the  auxiliary variable *aux_delay_weight* is set to equal the *log_base* value for all speeches before the shock.

# %%
df.loc[df.after_shock == 0, "aux_delay_weight"] = log_base

# %% [markdown]
# In the next chunk, the non-linear weighting value is completed. As all speeches before the shock have an *aux_delay_weight* equal to *log_base*, the final weight *delay_weight* will be $log_{log\_base}(log\_base) = 1$.
# Accordingly, more importance is attached to speeches in the first two plenary meetings after the external shock. A weight of 1 would be attached to all plenary meetings before the external shock and plenary meeting 3 after the shock. All plenary meetings after number 3 would receive a slowly decreasing importance. For example, meeting 200 after the shock would receive a weight of $\frac{1}{log_3 200}=\frac{1}{4.8}$ - the opinions voiced at that time are roughly a fifth as important as an opinion voiced in plenary meeting 3 after the shock.

# %%
df["delay_weight"] = df["aux_delay_weight"].apply(lambda x: log(x, log_base))

# %% [markdown]
# Lastly, all auxialilary columns no longer needed can be removed.

# %%
del df["delay"]
del df["aux_delay"]
del df["aux_delay_weight"]

# %% [markdown]
# #### Application of Delay Weights

# %%
df["w_score"] = df["score"] / df["delay_weight"]
df["w_score_p"] = df["score_p"] / df["delay_weight"]
df["w_score_a"] = df["score_a"] / df["delay_weight"]

# %% [markdown]
# ### Review of Preliminary Results
# The next chunk returns a sample of rows with strong positive and negative scores. These speech fragments are examples of extremely strong opinions towards nuclear energy politics being voiced in parliament.

# %%
with pd.option_context(
    "display.max_colwidth",
    25,
    "display.precision",
    2,
    "display.float_format",
    lambda x: "%.2f" % x,
):
    print(
        df[
            [
                "PN_s",
                "PN_sp",
                "PN_sa",
                "AN_s",
                "AN_sp",
                "AN_sa",
                "CE_s",
                "CE_sp",
                "CE_sa",
                "NE_s",
                "NE_sp",
                "NE_sa",
                "score",
                "score_p",
                "score_a",
                "w_score",
                "w_score_p",
                "w_score_a",
                "delay_weight",
            ]
        ].describe()
    )

# %%
