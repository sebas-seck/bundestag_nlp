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
#
# # Score Calculation
#
# The score calculation is to be part of `run_opinion_logic()` after development through the flag `compute_scores`.

# %%
import os
import pandas as pd
import numpy as np

from src.opinion_logic import run_opinion_logic, ALL_KEYWORDS_EXTENDED
from src.utils import show, parse_config
from src.data_prep import prepare_speech_data

CONFIG = parse_config()


# %%
# %%time


def process():
    print("Not using cached df, processing now")
    df = prepare_speech_data(ALL_KEYWORDS_EXTENDED)
    df = run_opinion_logic(df, compute_scores=False)
    # df = run_opinion_logic(df, subset_start=20000, subset_end=20100)
    return df


if CONFIG["use_cache"]:
    if os.path.exists(CONFIG["processed_df_cache"]):
        print("Using cached, previously processed dataframe")
        df = pd.read_pickle(CONFIG["processed_df_cache"])
    else:
        print("No cached data available")
        df = process()
else:
    df = process()

# %%
# query df on focus area index 2009-2015 here!
df.query("electoral_term  >= 17 and session >= 94 and electoral_term <= 19")

# %%
show(df)

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
# The delay weight ensures speeches around the external shock are accounted for with a higher weight than speeches long before or after. The delay weight takes values between 1 and 5.

# %%
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

# %%
df.plot.line(x="date", y="weight")

# %%
df["w_score"] = df["score"] / df["weight"]
df["w_score_p"] = df["score_p"] / df["weight"]
df["w_score_a"] = df["score_a"] / df["weight"]

# %% [markdown]
# ### Descriptive Analysis of assigned Scores

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
                "weight",
            ]
        ].describe()
    )

# %%
