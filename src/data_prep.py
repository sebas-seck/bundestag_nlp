# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd

from src.utils import parse_config

CONFIG = parse_config()


def prepare_data(filter_list):
    """
    Prepares Open Discourse speech data.

    - Filters for specified keywords
    - Renames columns
    - Replaces numbering and whitespace inside speeches
    - Adds flag whether a speech was before or after the external shock

    Parameters
    ----------
    filter_list : List
        List with keywords to be filtered for

    Returns
    -------
    pd.DataFrame
        Prepared Dataframe with all speeches that contain at least one keyword
    """
    df = pd.read_csv("data/open_discourse/speeches.csv", parse_dates=["date"])
    df.rename(columns={"speechContent": "text"}, inplace=True)
    print("Prior shape", df.shape)
    # drops speech entries without content
    print("Speech entries without content", sum(df["text"].isnull()))
    df.dropna(subset=["text"], inplace=True)
    # If executable, this filters all speeches and keeps only those which include at least one word from the hardcoded keywords later in the notebook
    df = df[df["text"].str.contains(" | ".join(filter_list))].copy()
    df = df.reset_index(drop=True)
    print("Shape after filter", df.shape)
    df["after_shock"] = np.where(
        df["date"] < pd.Timestamp(year=2011, month=3, day=11), False, True
    )
    df.rename(
        columns={
            "electoralTerm": "electoral_term",
            "firstName": "first_name",
            "lastName": "last_name",
            "politicianId": "politician_id",
            "factionId": "faction_id",
            "documentUrl": "document_url",
            "positionShort": "position_short",
            "positionLong": "position_long",
        },
        inplace=True,
    )

    def _replace_numbering(text):
        return re.sub(r"\(\{[0-9]+\}\)", "", text)

    def _replace_whitespace(text):
        return re.sub("\s+", " ", text)

    df["text"] = df["text"].replace(to_replace="\n", value=" ", regex=True)
    df["text"] = df["text"].apply(lambda x: _replace_numbering(x))
    df["text"] = df["text"].apply(lambda x: _replace_whitespace(x))

    df.to_pickle(CONFIG["df_prep_pickle_path"])

    return df
