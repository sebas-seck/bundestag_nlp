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

# %%
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
import pandas as pd
import glob


# %%
def show(table):
    with pd.option_context("display.max_colwidth", None):
        display(table)


# %%
path = "plpr-scraper/data/out"
all_files = glob.glob(path + "/*.csv")

df_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, escapechar="\\")
    df_list.append(df)

df = pd.concat(df_list, axis=0, ignore_index=True)
df.text = df.text.str.replace("\n", " ")
show(df[:50])

# %%
