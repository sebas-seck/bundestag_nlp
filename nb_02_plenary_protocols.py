# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plenary Protocols

# %%
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
from pathlib import Path
import glob
import pandas as pd

# %%
VERBOSE = False


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
df = df[~df["text"].isnull()].copy()
df.loc[:, "text"] = df.loc[:, "text"].str.replace("\n\n", " ").replace("\n", " ")
df.loc[:, "text"] = df.loc[:, "text"].astype(str)
df = df[df["type"] == "speech"]
df = df.reset_index(drop=True)
df.drop(["Unnamed: 0", "row.names"], axis=1, inplace=True, errors="ignore")
print(f"The shape is reduced from {df_prior_shape[0]} rows to {df.shape[0]}")
show(df[:3])

# %%
show(df[df["text"].str.contains("\. nan")][:5])

# %% [markdown]
#
#
# ### Option A: Slow but sizable junks without mid-sentence interruptions
# It's slow but sizable junks without mid-sentence interruptions. Speeches span multiple rows, such cases can be joined partly, to finish on full sentences. A loop is computationally expensive (~25 min) but does the work!

# %%
# %%time
path_plpr_a = "data/plpr_a.pkl"
if Path(path_plpr_a).exists():
    df_a = pd.read_pickle(path_plpr_a)

else:
    df_a = df.copy().reset_index(drop=True)
    for i in range(0, df_a.shape[0] - 1):
        if df_a.at[i, "speaker_fp"] == df_a.at[i + 1, "speaker_fp"]:
            # if type(df_a.at[i,'text']) == float:
            # print(df_a.at[i,'text'])
            if df_a.at[i, "text"].endswith("." or "!" or "?" or ":"):
                continue
            else:
                df_a.at[i + 1, "text"] = str(
                    df_a.at[i, "text"] + " " + df_a.at[i + 1, "text"]
                )
                df_a.drop(i, inplace=True)
            # stop the transformation as the loop runs out of index

    df_a = df_a.reset_index(drop=True)  # reindexing
    df_a.to_pickle(path_plpr_a)
show(df_a[:3])

# %% [markdown]
# ### Option B: Fast per speech with some column detail disregarded
#

# %%
for name in df.columns:
    print(df.at[0, name] == df.at[1, name], name)

# %%
# %%time
if 1 == 1:
    df_b = df.copy()
    df_b["speaker_fp_duplicate"] = df_b["speaker_fp"].copy()
    adjacent_rows_grouper = (df_b["speaker_fp"].shift() != df_b["speaker_fp"]).cumsum()
    df_b["id"] = df_b["id"].astype(str)
    df_b = (
        df_b.groupby(
            [
                adjacent_rows_grouper,
                "sitzung",
                "wahlperiode",
                "speaker_cleaned",
                "filename",
                "type",
            ]
        )["text"]
        .apply(" ".join)
        .reset_index()
    )
    df_b.to_pickle("data/plpr_b.pkl")
    show(df_b[:5])

# %% [markdown]
# ### All Text
# To model topics, metainformation for speeches is not relevant. Everything in the text column can be glued together for that purpose.

# %%
# Option A
path_alltext_a = Path("data/plpr_alltext_a.txt")

df_a["text"].to_csv(path_alltext_a, sep=" ", index=False, header=False)
preview_lines(path_alltext_a, N=2)

# %%
# Option B
path_alltext_b = Path("data/plpr_alltext_b.txt")

# df_b["text"] = df_b["text"].str.replace(". nan", ".")
# df_b["text"][df_b["text"] != "nan"].to_csv(path_alltext_b, sep=" ", index=False, header=False)
df_b["text"].to_csv(path_alltext_b, sep=" ", index=False, header=False)

# %%
