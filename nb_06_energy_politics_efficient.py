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
# # External Shock on Energy Politics with new source

# %%
import time
from math import log
import os

import numpy as np
import pandas as pd
import spacy
import stylecloud

from src.keywords import ANTI_NUCLEAR, CONSERVATIVE_ENERGY, NEUTRAL_ENERGY, PRO_NUCLEAR

# %%
DF_AVAIL = os.path.exists("df_prep.pkl")

# %% [markdown]
# ## Energy Politics Keywords
# The keyword lists from `src/keywords.py` are handcrafted with the help of the topic models in notebook 03. Assignment to a category is guided by these questions:
#
# - Is <keyword> helping to leave nuclear energy (in a sustainable manner)? If yes, the topic is **anti nulcear**.
# - Is <keyword> helping to keep nuclear energy? If yes, the topic is **pro nuclear**.
# - Is <keyword> associated with conservative energy and does not fit into the above categories? If yes, the topic is **conservative energy**.
# - Is <keyword> directly associated with energy politics but does not fit into the above categories? If yes, the topic is **neutral energy**.
#
# ### Stylecloud with keywords

# %%
text = " ".join(ANTI_NUCLEAR + CONSERVATIVE_ENERGY + NEUTRAL_ENERGY + PRO_NUCLEAR)
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
TOPICS = [ANTI_NUCLEAR, PRO_NUCLEAR, NEUTRAL_ENERGY, CONSERVATIVE_ENERGY]


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


# %% [markdown]
# ### Dataframe Preparation
# - filter out all speeches not related to energy politics
# - create a dummy variable whether a speech was given before or after the external shock

# %%
gerNLP = spacy.load("de_core_news_lg")

# %%
# applies topic extension
for i in range(0, len(TOPICS)):
    topics_extension(TOPICS[i])

# %%
ALL_KEYWORDS = ANTI_NUCLEAR + PRO_NUCLEAR + NEUTRAL_ENERGY + CONSERVATIVE_ENERGY

# %%
# %%time
if not DF_AVAIL:
    df = pd.read_csv("data/open_discourse/speeches.csv", parse_dates=["date"])
    df.rename(columns={"speechContent": "text"}, inplace=True)
    print("Prior shape", df.shape)
    # drops speech entries without content
    print("Speech entries without content", sum(df["text"].isnull()))
    df.dropna(subset=["text"], inplace=True)
    # If executable, this filters all speeches and keeps only those which include at least one word from the hardcoded keywords later in the notebook
    df = df[df["text"].str.contains("|".join(ALL_KEYWORDS))].copy()
    df = df.reset_index(drop=True)
    print("Shape after filter", df.shape)
    df["after_shock"] = np.where(
        df["date"] < pd.Timestamp(year=2011, month=3, day=11), False, True
    )
    df.to_pickle("df_prep.pkl")

# %% [markdown]
# The column *after_shock* is a dummy variable to indicate whether a speech fragment is part of a plenary meeting before the catastrophy in Fukushima or thereafter. The inflection point is between the plenary meetings 97 and 98 during the 17th legislative period.

# %% [markdown]
# ### Opinion Analysis Algorithm
#
# The class `OpinionAnalyzer()` implements
#
# - columns to store keywords and their sentiments as well as numeric scores
# - counters to track occurances of cases
# - method `calc_scores` takes a list of words and a list of associated negations and returns a total sentiment score of the list and a list documenting the score calculation

# %% [markdown]
# ## Algorithm Execution

# %%
df = pd.read_pickle("df_prep.pkl")

# %%
from src.opinion_logic import OpinionAnalyzer

# %%
# %%time

df = pd.read_pickle("df_prep.pkl")
opinion = OpinionAnalyzer(df)
df = opinion.main(batch_size=2, n_process=6)
opinion.protocol

# %% [markdown]
# Sample processing performance of 500 documents shown in table below.
#
# | batch_size | n_process | wall time | cpu time |
# |-|-|-|-|
# | 200 | 2 | 2h 14min | 1h 8min |
# | 20 | 6 | 1h 33min | 1h 5min |
# | 2 | 8 | 1h 6min | 1h 8min |
# | 2 | 12 | 1h 8min | 1h 4min |
# | 1 | 12 | 1h 53min | 1h 15min |
# | 5 | 12 | 3h 24min | 3h 21min |
# | 1 | 24 | 4h 29 min | 4h 18min |
# | 4 | 8 | 1h 26min | 1h 7min |
# | 2 | 6 | 1h 7min | 1h 6min |
# | 2 | 4 | 1h 55min | 1h 20min |
# | 2 | 2 | 1h 29min | 1h 28min |
# | 2 | 1 | 1h 29min | 1h 29min |
# | 18 | 2 | 1h 5min | 1h 5min |
# | 2 | 18 | 1h 46min | 1h 7min |
# | 15 | 4 | 1h 13min | 1h 11min |
# | 4 | 8 | 1h 26min | 1h 7min |
# | 2 | 18 | 1h 46min | 1h 7min |

# %%
import pickle

with open("opinion.pkl", "wb") as f:
    pickle.dump(opinion, f)

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
# The main score is calculated using three of the four keyword lists. The score is designed to be positive to reflect progressiveness. Therefore, anti-nuclear energy opinions are added, pro-nuclear energy opinions are substracted, and conservative energy politics opinions are substracted as well, as those do not reflect the turnaround performed by politics. Solely opinions about energy politics which do not fall into any of the other three categories are not included in the score but are kept for reference.

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
df.loc[df.electoralTerm <= 17, "aux_delay"] = -94
# for speeches given in legislative period 18, 159 is added as the external shock took place 159 plenary meetings before the first meeting in legislative period 18
df.loc[df.electoralTerm >= 18, "aux_delay"] = 253 - 94
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
