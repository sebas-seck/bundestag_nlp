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

# %%
import pandas as pd
import plotly.express as px

from src.utils import show, parse_config

CONFIG = parse_config()

# %%
df = pd.read_pickle(CONFIG["df_processed_pickle_path"])

df_faction = pd.read_csv("data/open_discourse/factions.csv")
df_faction.rename(
    columns={
        "id": "faction_id",
        "abbreviation": "faction_abbreviation",
        "fullName": "faction_name",
    },
    inplace=True,
)
df_faction.index.rename("index", inplace=True)

# %%
df_faction

# %%
df_new = pd.merge(
    df, df_faction, how="left", left_on="faction_id", right_on="faction_id"
)


# %%
party_colors = {
    "CDU/CSU": "black",
    "SPD": "red",
    "Grüne": "green",
    "FDP": "yellow",
    "DIE LINKE.": "magenta",
    "PDS": "magenta",
    "DRP/NR": "grey",
    "KO": "grey",
    "SSW": "grey",
    "WAV": "grey",
    "BHE": "grey",
    "DPB": "grey",
    "Fraktionslos": "grey",
    "FU": "grey",
    "FVP": "grey",
    "BP": "grey",
    "NR": "grey",
    "DA": "grey",
    "not found": "grey",
}

# %%
df_new.dropna(subset=["faction_abbreviation"], inplace=True)

# %%
fig = px.scatter(
    df_new.query("w_score != 0"),
    x="date",
    y="w_score",
    color="faction_abbreviation",
    color_discrete_map=party_colors,
    size="speech_length",
    opacity=0.5,
    title="Speeches on Energy Politics",
    hover_data=["last_name", "faction_id"],
)
fig.show()

# %%
fig.write_html("nb_07_energy_politics_timeline.html")
