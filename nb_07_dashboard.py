# -*- coding: utf-8 -*-

# Run this app with `python nb_07_dashboard.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

from src.data_prep import prepare_full_df
from src.utils import parse_config

CONFIG = parse_config()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

df = prepare_full_df()

df["year"] = df["date"].dt.year

fig = px.scatter(
    df.query("w_score != 0"),
    x="date",
    y="w_score",
    color="faction_abbreviation",
    color_discrete_map=CONFIG["party_colors"],
    size="speech_length",
    opacity=0.5,
    title="Speeches on Energy Politics",
    custom_data=["full_name", "profession"],
    labels={"faction_abbreviation": "Faction"},
)
fig.update_traces(hovertemplate="%{customdata[0]}<br>%{customdata[1]}")

app.layout = html.Div(
    children=[
        html.H1(children="Timeline of Energy Politics"),
        #     html.Div(
        #         children="""
        #     Dash: A web application framework for Python.
        # """
        #     ),
        dcc.Graph(id="timeline-bubbles-with-slider", figure=fig),
        dcc.RangeSlider(
            id=" range-slider",
            min=df["year"].min(),
            max=df["year"].max(),
            value=[2000, 2020],
            step=1,
            marks={year: str(year) for year in list(range(1950, 2021, 5))},
        ),
        html.Div(id="speech-drilldown", children="")
        #     dcc.Textarea(
        #     id='speech-drilldown',
        #     # value='Textarea content initialized\nwith multiple lines of text',
        #     value='',
        #     style={'width': '100%', 'height': 300},
        # ),
    ]
)
import random


@app.callback(
    Output("speech-drilldown", "children"),
    Input("timeline-bubbles-with-slider", "clickData"),
)
def display_click_data(selectedData):
    if selectedData is None:
        return ""
    speech_index = selectedData["points"][0]["customdata"][0]
    return f"{df.loc[speech_index, 'text']}"


@app.callback(
    Output("timeline-bubbles-with-slider", "figure"), Input(" range-slider", "value")
)
def update_figure(range):
    fig = px.scatter(
        df.query(f"year >= {range[0]} and year <= {range[1]} and w_score != 0"),
        x="date",
        y="w_score",
        color="faction_abbreviation",
        color_discrete_map=CONFIG["party_colors"],
        size="speech_length",
        opacity=0.5,
        title=f"Speeches on Energy Politics (selected range {range[0]}-{range[1]})",
        custom_data=["index", "full_name", "profession", "faction_name"],
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

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
