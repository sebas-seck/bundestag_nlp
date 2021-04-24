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
# # Plenary Protocols

# %%
from pathlib import Path
import glob
import numpy as np
import pandas as pd

# %%
VERBOSE = True


# %%
def show(table):
    with pd.option_context("display.max_colwidth", None):
        display(table)


def preview_lines(filepath, N=5):
    with open(filepath) as temp:
        head = [next(temp) for i in range(N)]
    temp.close()
    return head


# %% [markdown]
# ### Data Loading
# All csv-files in the specified directory will be parsed and concatenated.

# %%
path = "plpr-scraper/data/out"
all_files = glob.glob(path + "/*.csv")

df_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, escapechar="\\")
    df_list.append(df)

df = pd.concat(df_list, axis=0, ignore_index=True)
df_prior_shape = df.shape
show(df[:10])

# %% [markdown]
# ### Data Cleansing
# Quick look at the data before cleansing with Pandas-Profiling. We've got 592199 observations, many of which are not speeches but contributions by the *chair* (president and vice-presidents of the parliament) and *POIs*, points of information (interruptions or interjections by MPs other than the speaker or chair). Speeches span multiple rows and require concatenation.

# %%
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Pre-Cleansing Pandas Profiling Report")

# %%
if VERBOSE:
    profile.to_notebook_iframe()

# %% [markdown]
# For now, we'll keep only speeches and replace newlines read-in literally.

# %%
df.dropna(subset=["text"], inplace=True)
df.loc[:, "text"] = df.loc[:, "text"].str.replace("\n\n", " ").replace("\n", " ")
df.loc[:, "text"] = df.loc[:, "text"].astype(str)
df = df[df["type"] == "speech"].copy()
df.reset_index(drop=True, inplace=True)
df.drop(["Unnamed: 0", "row.names"], axis=1, inplace=True, errors="ignore")
print(f"The shape is reduced from {df_prior_shape[0]} rows to {df.shape[0]}")
show(df[["speaker_cleaned", "text"]][:3])

# %% [markdown]
# todo describe what happens

# %%
df["previous_speaker_fp"] = df["speaker_fp"].shift(1)
df["new_speaker"] = df["speaker_fp"] != df["previous_speaker_fp"]
df["speech_identifier"] = np.nan
df

# %%
# %%time
speech_identifier = int(0)
for index, row in df.iterrows():
    if row["new_speaker"]:
        speech_identifier += 1
    df.at[index, "speech_identifier"] = speech_identifier

# %%
df = (
    df.groupby(["speaker_fp", "speech_identifier"], sort=False)
    .agg(
        {
            "id": "count",
            "sitzung": "first",
            "wahlperiode": "first",
            "speaker": "first",
            "speaker_cleaned": "first",
            "sequence": min,
            "text": " ".join,
            "filename": "first",
            "type": "first",
            "speaker_party": "first",
        }
    )
    .reset_index()
)

# %%
df.to_pickle("data/plpr.pkl")

# %% [markdown]
# ### All Text
# To model topics, metainformation for speeches is not relevant. Everything in the text column can be glued together for that purpose.

# %%
path_alltext = Path("data/plpr_alltext.txt")

df["text"].to_csv(path_alltext, sep=" ", index=False, header=False)
preview_lines(path_alltext, N=2)

# %%
