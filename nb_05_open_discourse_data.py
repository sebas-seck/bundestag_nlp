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

# %%
import pandas as pd

# %%
df = pd.read_csv("data/open_discourse/contributions_extended.csv", nrows=100)
df

# %%
df = pd.read_csv("data/open_discourse/contributions_simplified.csv", nrows=100)
df

# %%
df = pd.read_csv("data/open_discourse/factions.csv")
df

# %%
df = pd.read_csv("data/open_discourse/politicians.csv")
df

# %%
df = pd.read_csv("data/open_discourse/speeches.csv", nrows=10, parse_dates=["date"])
df

# %%
df["positionShort"].unique()

# %%
