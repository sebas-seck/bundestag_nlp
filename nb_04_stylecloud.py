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
import stylecloud

from src.keywords import ANTI_NUCLEAR, CONSERVATIVE_ENERGY, NEUTRAL_ENERGY, PRO_NUCLEAR

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
