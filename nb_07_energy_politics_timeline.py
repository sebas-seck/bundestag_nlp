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
from src.data_prep import prepare_faction_data, prepare_politician_data

CONFIG = parse_config()

# %%
# speech df already fully processed
df_speeches = pd.read_pickle(CONFIG["processed_df_cache"])
df_faction = prepare_faction_data()
df_faction

# %%
df = pd.merge(
    df_speeches, df_faction, how="left", left_on="faction_id", right_on="faction_id"
)


# %%
df_politicians = prepare_politician_data()
df_politicians

# %%
df = pd.merge(
    df, df_politicians, how="left", on="politician_id", suffixes=("_speech", "_master")
)
df

# %%
df = df.dropna(subset=["faction_abbreviation"])

# %%
fig = px.scatter(
    df.query("w_score != 0"),
    x="date",
    y="w_score",
    color="faction_abbreviation",
    color_discrete_map=CONFIG["party_colors"],
    size="speech_length",
    opacity=0.5,
    title=f"Speeches on Energy Politics",
    custom_data=["index", "full_name_speech", "profession", "faction_name"],
    labels={"faction_abbreviation": "Faction"},
    category_orders={
        "faction_abbreviation": [
            "SPD",
            "CDU/CSU",
            "Grüne",
            "FDP",
            "DIE LINKE.",
            "AfD",
            "PDS",
            "Fraktionslos",
            "not found",
        ]
    },
)
fig.update_traces(
    hovertemplate="%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}"
)
fig.show()

# %%
fig.write_html("nb_07_energy_politics_timeline.html")

# %%
