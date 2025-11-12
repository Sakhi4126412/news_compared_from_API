from difflib import get_close_matches
import pandas as pd

def normalize_rating(rating):
    mapping = {
        "True": 1.0, "Mostly True": 0.8, "Half True": 0.5,
        "Mixture": 0.5, "Mostly False": 0.2, "False": 0.0,
        "Unknown": None, "Unproven": 0.4, "Legend": 0.3,
        "Miscaptioned": 0.3, "Outdated": 0.4
    }
    for key in mapping:
        if key.lower() in rating.lower():
            return mapping[key]
    return None


def match_claims(df_snopes, df_politifact):
    matched = []
    for _, row in df_snopes.iterrows():
        matches = get_close_matches(row["claim"], df_politifact["claim"].tolist(), n=1, cutoff=0.4)
        if matches:
            match_row = df_politifact[df_politifact["claim"] == matches[0]].iloc[0]
            matched.append({
                "claim_snopes": row["claim"],
                "rating_snopes": row["rating"],
                "rating_snopes_num": row["rating_numeric"],
                "claim_politifact": match_row["claim"],
                "rating_politifact": match_row["rating"],
                "rating_politifact_num": match_row["rating_numeric"]
            })
    return pd.DataFrame(matched)
