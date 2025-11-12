"""
app/utils.py
Helper functions for fact-check comparison app.
Includes rating normalization and claim matching utilities.
"""

import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher


def normalize_rating(rating: str) -> float:
    """
    Normalize textual ratings from various fact-checkers into a numeric scale [0, 1].
    1.0 = completely true
    0.0 = completely false
    """
    if not isinstance(rating, str):
        return np.nan

    text = rating.strip().lower()

    # Common true/false labels
    mapping = {
        # True/Mostly True
        "true": 1.0,
        "mostly true": 0.9,
        "largely true": 0.9,
        "accurate": 0.9,
        "correct": 1.0,
        "verified": 1.0,
        "supported": 0.8,

        # Half true or mixed
        "half true": 0.5,
        "partly true": 0.5,
        "mixed": 0.5,
        "partially true": 0.5,
        "unproven": 0.5,
        "unsubstantiated": 0.4,
        "unclear": 0.5,
        "disputed": 0.5,

        # Mostly false or misleading
        "mostly false": 0.2,
        "barely true": 0.3,
        "misleading": 0.2,
        "unsupported": 0.3,
        "exaggerated": 0.3,
        "false": 0.0,
        "wrong": 0.0,
        "incorrect": 0.0,
        "pants on fire": 0.0,
        "fiction": 0.0,
        "no evidence": 0.1,
        "fake": 0.0
    }

    # Find the closest match in mapping
    for key, val in mapping.items():
        if key in text:
            return val

    # Default if not found
    return np.nan


def match_claims(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """
    Match claims from two fact-checker datasets using approximate string similarity.

    Args:
        df1, df2: DataFrames with a 'claim' column.
        threshold: minimum ratio for considering two claims a match.

    Returns:
        Merged DataFrame containing matched claims and both numeric ratings.
    """
    matches = []
    for _, row1 in df1.iterrows():
        claim1 = str(row1.get("claim", "")).strip()
        best_match = None
        best_score = 0.0
        for _, row2 in df2.iterrows():
            claim2 = str(row2.get("claim", "")).strip()
            ratio = SequenceMatcher(None, claim1, claim2).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = row2

        if best_score >= threshold and best_match is not None:
            matches.append({
                "claim": claim1,
                "rating_source1": row1.get("rating_numeric"),
                "rating_source2": best_match.get("rating_numeric"),
                "similarity": best_score
            })

    return pd.DataFrame(matches)


def clean_claim_text(text: str) -> str:
    """
    Remove unnecessary punctuation, quotes, and whitespace from claim text.
    Useful for standardizing claim text before matching.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip().lower()
