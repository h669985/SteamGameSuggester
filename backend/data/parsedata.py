import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

PARQUET_PATH = "hf://datasets/FronkonGames/steam-games-dataset/data/train-00000-of-00001-e2ed184370a06932.parquet"

# ---------- helpers ----------
def _as_list(v):
    """Parse list-ish strings ('A,B', '["A","B"]') into Python lists."""
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        # JSON array?
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return j
            except Exception:
                pass
        # comma-separated
        if "," in s:
            parts = [t.strip() for t in s.split(",")]
            return [p for p in parts if p]
        # single token string
        return [s] if s else []
    # unknown type
    return [str(v)]

def _parse_estimated_owners(s):
    """'0 - 20000' -> (0, 20000, 10000)"""
    if not s or not isinstance(s, str):
        return None, None, None
    m = re.findall(r"\d[\d,]*", s)
    if len(m) >= 2:
        low = int(m[0].replace(",", ""))
        high = int(m[1].replace(",", ""))
        mid = (low + high) // 2
        return low, high, mid
    return None, None, None

def _parse_release_year(s):
    """Try to extract a release year from strings like 'Oct 21, 2008' or '2015'."""
    if not s:
        return None
    # direct year
    m = re.search(r"(19|20)\d{2}", str(s))
    if m:
        return int(m.group(0))
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.notna(dt):
            return int(dt.year)
    except Exception:
        pass
    return None

def _snake(name):
    return re.sub(r"\W+", "_", name).strip("_").lower()

# ---------- load & normalize ----------
def load_df(path=PARQUET_PATH):
    df = pd.read_parquet(path)
    df = df.replace({np.nan: None})
    return df

def build_canonical(df: pd.DataFrame):
    # Column names from your schema (exact case)
    # ID
    id_col = "AppID"

    # UI label columns (kept for picker display only)
    ui_cols = ["Name", "Price", "Metacritic score", "User score", "Positive", "Negative", "Genres"]

    # Numeric/boolean features you want to feed the model
    numeric_bool_cols = [
        "Price",
        "User score",
        "Metacritic score",
        "Positive",
        "Negative",
        "Peak CCU",
        "Achievements",
        "DLC count",
        "Average playtime forever",
        "Average playtime two weeks",
        "Median playtime forever",
        "Median playtime two weeks",
        "Recommendations",
        "Required age",
        "Score rank",
        "Windows",
        "Mac",
        "Linux",
    ]

    # Engineered features from raw columns
    engineer_from = {
        "Release date": "release_year",
        "Estimated owners": ("owners_low", "owners_high", "owners_mid"),
    }

    # Categorical multi-valued fields to keep as lists (you’ll encode later)
    listish_cols = ["Developers", "Publishers", "Categories", "Genres", "Tags"]

    # --- Build dataset (UI) ---
    dataset = {}
    for _, row in df[[id_col] + [c for c in ui_cols if c in df.columns]].iterrows():
        rid = str(row[id_col])
        rec = {
            "name": row.get("Name"),
            "price": row.get("Price"),
            "metacritic_score": row.get("Metacritic score"),
            "user_score": row.get("User score"),
            "positive": row.get("Positive"),
            "negative": row.get("Negative"),
            "genres": _as_list(row.get("Genres")),
        }
        dataset[rid] = rec

    # --- Build features_df ---
    # Start with numeric/boolean (rename to snake_case)
    feat_parts = []
    nb_present = [c for c in numeric_bool_cols if c in df.columns]
    nb = df[[id_col] + nb_present].copy()

    rename_map = {c: _snake(c) for c in nb_present}
    nb.rename(columns=rename_map, inplace=True)

    # coerce numeric types sensibly
    for col in [
        "price",
        "user_score",
        "metacritic_score",
        "positive",
        "negative",
        "peak_ccu",
        "achievements",
        "dlc_count",
        "average_playtime_forever",
        "average_playtime_two_weeks",
        "median_playtime_forever",
        "median_playtime_two_weeks",
        "recommendations",
        "required_age",
        "score_rank",
    ]:
        if col in nb.columns:
            nb[col] = pd.to_numeric(nb[col], errors="coerce")

    # booleans: ensure True/False not None
    for col in ["windows", "mac", "linux"]:
        if col in nb.columns:
            nb[col] = nb[col].fillna(False).astype(bool)

    feat_parts.append(nb.set_index(id_col))

    # Engineered: release_year
    if "Release date" in df.columns:
        release_year = df[["AppID", "Release date"]].copy()
        release_year["release_year"] = release_year["Release date"].map(_parse_release_year)
        release_year = release_year.drop(columns=["Release date"]).set_index("AppID")
        feat_parts.append(release_year)

    # Engineered: estimated owners (low, high, mid)
    if "Estimated owners" in df.columns:
        owners = df[["AppID", "Estimated owners"]].copy()
        owners[["owners_low", "owners_high", "owners_mid"]] = owners["Estimated owners"].apply(
            lambda s: pd.Series(_parse_estimated_owners(s))
        )
        owners = owners.drop(columns=["Estimated owners"]).set_index("AppID")
        feat_parts.append(owners)

    # Categorical list-ish fields: keep as lists (object dtype)
    cat = {}
    for c in listish_cols:
        if c in df.columns:
            cat[_snake(c)] = df[c].map(_as_list)
    if cat:
        cat_df = pd.DataFrame(cat)
        cat_df.index = df[id_col].astype(str)
        feat_parts.append(cat_df)

    # Concatenate all parts (align on AppID)
    if feat_parts:
        features_df = pd.concat(feat_parts, axis=1)
    else:
        features_df = pd.DataFrame()

    # Finalize: index must be string AppID
    features_df.index = df[id_col].astype(str)

    # Optional: stable column order (numeric/booleans first, engineered, then categoricals)
    ordered_cols = (
        [c for c in nb.set_index(id_col).columns if c in features_df.columns]
        + [c for c in ["release_year", "owners_low", "owners_high", "owners_mid"] if c in features_df.columns]
        + [c for c in [_snake(c) for c in listish_cols] if c in features_df.columns]
    )
    features_df = features_df[ordered_cols]

    return dataset, features_df

def build_features(app_ids, features_df):
    """Return the feature rows for the given list of AppID strings, in the exact training format."""
    if not isinstance(app_ids, (list, tuple, set)):
        app_ids = [app_ids]
    app_ids = [str(x) for x in app_ids]
    # keep only ids present
    ids = [i for i in app_ids if i in features_df.index]
    return features_df.loc[ids]

# --------- module outputs ---------
_df = load_df(PARQUET_PATH)
dataset, features_df = build_canonical(_df)
