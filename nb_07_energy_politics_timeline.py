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
    "Fraktionslos": "lightgrey",
    "DRP/NR": "lightgrey",
    "KO": "lightgrey",
    "SSW": "lightgrey",
    "WAV": "lightgrey",
    "BHE": "lightgrey",
    "DPB": "lightgrey",
    "FU": "lightgrey",
    "FVP": "lightgrey",
    "BP": "lightgrey",
    "NR": "lightgrey",
    "DA": "lightgrey",
    "not found": "lightgrey",
}

# %%
df_new.dropna(subset=["faction_abbreviation"], inplace=True)

# %%
df_new["first_name"]

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
    custom_data=["first_name", "last_name", "faction_name"],
    labels={"faction_abbreviation": "Faction"},
)
fig.update_traces(hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}")
fig.show()

# %%
fig.write_html("nb_07_energy_politics_timeline.html")
