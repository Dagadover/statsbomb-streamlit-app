from pathlib import Path
from typing import Dict, Union, Any
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# =========================
# Constants (StatsBomb)
# =========================
PITCH_L = 120.0
PITCH_W = 80.0
GOAL_Y = 40.0


# =========================
# Normalizers
# =========================
def to_name(v: Any) -> Any:
    """
    If v is a dict-like object containing 'name', return v['name'].
    Otherwise return v unchanged.
    """
    if isinstance(v, dict) and "name" in v:
        return v["name"]
    return v


def normalize_match_row(match_row: pd.Series) -> Dict[str, Union[str, int, float]]:
    """
    Extract match metadata robustly even if home_team/away_team come as dicts.
    """
    def pick(*cols, default=None):
        for c in cols:
            if c in match_row.index:
                return match_row[c]
        return default

    match_id = int(pick("match_id", default=-1))

    home_raw = pick("home_team", "home_team_name", default="")
    away_raw = pick("away_team", "away_team_name", default="")

    home_team = to_name(home_raw)
    away_team = to_name(away_raw)

    home_score = pick("home_score", default=np.nan)
    away_score = pick("away_score", default=np.nan)

    try:
        home_score = float(home_score)
        away_score = float(away_score)
    except Exception:
        home_score, away_score = np.nan, np.nan

    return {
        "match_id": match_id,
        "home_team": str(home_team),
        "away_team": str(away_team),
        "home_score": float(home_score),
        "away_score": float(away_score),
        "goal_diff": float(home_score - away_score)
        if np.isfinite(home_score) and np.isfinite(away_score)
        else np.nan,
    }


def normalize_event_columns(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'type' and 'team' exist and are strings.
    Also convert dicts like {'id':..,'name':'Pass'} to 'Pass'.
    """
    ev = ev.copy()

    if "type" not in ev.columns and "type_name" in ev.columns:
        ev["type"] = ev["type_name"]
    if "team" not in ev.columns and "team_name" in ev.columns:
        ev["team"] = ev["team_name"]

    if "type" in ev.columns:
        ev["type"] = ev["type"].apply(to_name).astype(str)
    if "team" in ev.columns:
        ev["team"] = ev["team"].apply(to_name).astype(str)

    return ev


# =========================
# Core preprocessing
# =========================
def prep_events_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract x,y from location and pass_end_x,pass_end_y from pass_end_location.
    Supports list/tuple/np.ndarray and stringified lists.
    """
    df = df.copy()

    def to_xy(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan, np.nan

        if isinstance(v, np.ndarray):
            v = v.tolist()

        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return v[0], v[1]

        if isinstance(v, str):
            s = v.strip()
            try:
                parsed = eval(s, {"__builtins__": {}})
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    return parsed[0], parsed[1]
            except Exception:
                return np.nan, np.nan

        return np.nan, np.nan

    if "location" in df.columns:
        xy = df["location"].apply(to_xy)
        df["x"] = xy.apply(lambda t: t[0])
        df["y"] = xy.apply(lambda t: t[1])

    if "pass_end_location" in df.columns:
        exy = df["pass_end_location"].apply(to_xy)
        df["pass_end_x"] = exy.apply(lambda t: t[0])
        df["pass_end_y"] = exy.apply(lambda t: t[1])

    return df


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror coordinates so away team attacks to the right.
    """
    df = df.copy()
    df["x"] = PITCH_L - df["x"]
    df["y"] = PITCH_W - df["y"]
    if "pass_end_x" in df.columns:
        df["pass_end_x"] = PITCH_L - df["pass_end_x"]
        df["pass_end_y"] = PITCH_W - df["pass_end_y"]
    return df


def detect_xg_col(df: pd.DataFrame):
    for c in ["shot_statsbomb_xg", "shot_xg", "statsbomb_xg"]:
        if c in df.columns:
            return c
    return None


def open_play_completed_passes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completed open-play passes.
    """
    df = df.copy()

    if "pass_outcome" in df.columns:
        df["pass_outcome"] = df["pass_outcome"].apply(to_name)
        df = df[df["pass_outcome"].isna()]

    banned = {"corner", "free kick", "throw-in", "goal kick", "kick off"}
    for c in ["pass_type", "pass_type_name"]:
        if c in df.columns:
            df[c] = df[c].apply(to_name)
            df = df[~df[c].astype(str).str.lower().isin(banned)]
            break

    return df


# =========================
# Feature extraction
# =========================
def shots_xg(ev, team):
    xg_col = detect_xg_col(ev)
    shots = ev[(ev["type"] == "Shot") & (ev["team"] == team)].copy()

    if shots.empty:
        return 0, np.nan

    xg = shots[xg_col].sum() if xg_col else np.nan
    return int(len(shots)), float(xg) if np.isfinite(xg) else np.nan


def progressive_passes(ev, team, is_away: bool):
    p = ev[(ev["type"] == "Pass") & (ev["team"] == team)].copy()
    if p.empty:
        return 0, 0.0

    p = open_play_completed_passes(p)

    for c in ["x", "y", "pass_end_x", "pass_end_y"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")

    p = p.dropna(subset=["x", "y", "pass_end_x", "pass_end_y"])
    if p.empty:
        return 0, 0.0

    if is_away:
        p = mirror(p)

    start = np.sqrt((PITCH_L - p["x"]) ** 2 + (GOAL_Y - p["y"]) ** 2)
    end = np.sqrt((PITCH_L - p["pass_end_x"]) **
                  2 + (GOAL_Y - p["pass_end_y"]) ** 2)
    prog = start - end
    prog = prog[prog >= 10]

    return int(len(prog)), float(prog.sum()) if len(prog) else 0.0


def pressures(ev, team, is_away: bool):
    pr = ev[(ev["type"] == "Pressure") & (ev["team"] == team)].copy()
    if pr.empty:
        return 0, 0

    pr["x"] = pd.to_numeric(pr["x"], errors="coerce")
    pr = pr.dropna(subset=["x"])
    if pr.empty:
        return 0, 0

    if is_away:
        pr = mirror(pr)

    mid = ((pr["x"] >= 40) & (pr["x"] < 80)).sum()
    fin = (pr["x"] >= 80).sum()
    return int(mid), int(fin)


def extract_match(events_path: Path, match_path: Path) -> Dict[str, Union[int, float, str]]:
    ev = pd.read_parquet(events_path)
    match_df = pd.read_parquet(match_path)

    match_row = match_df.iloc[0]
    meta = normalize_match_row(match_row)

    ev = prep_events_for_plots(ev)
    ev = normalize_event_columns(ev)

    home = meta["home_team"]
    away = meta["away_team"]

    sh_h, xg_h = shots_xg(ev, home)
    sh_a, xg_a = shots_xg(ev, away)

    pp_h, ppm_h = progressive_passes(ev, home, is_away=False)
    pp_a, ppm_a = progressive_passes(ev, away, is_away=True)

    pm_h, pf_h = pressures(ev, home, is_away=False)
    pm_a, pf_a = pressures(ev, away, is_away=True)

    return {
        "match_id": meta["match_id"],
        "home_team": home,
        "away_team": away,
        "goal_diff": meta["goal_diff"],

        "xg_diff": (xg_h - xg_a) if (np.isfinite(xg_h) and np.isfinite(xg_a)) else np.nan,
        "shots_diff": sh_h - sh_a,
        "prog_passes_diff": pp_h - pp_a,
        "prog_total_m_diff": ppm_h - ppm_a,
        "press_mid_diff": pm_h - pm_a,
        "press_fin_diff": pf_h - pf_a,
    }


# =========================
# Dataset builder
# =========================
def build_dataset(root_dir: str) -> pd.DataFrame:
    rows = []
    for d in Path(root_dir).glob("statsbomb_match_*"):
        try:
            row = extract_match(d / "events.parquet", d / "match.parquet")
            rows.append(row)
        except Exception as e:
            print(f"Skipping {d.name}: {e}")

    if not rows:
        raise RuntimeError("No matches loaded")

    return pd.DataFrame(rows)


# =========================
# Train & evaluate + SAVE MODEL
# =========================
def train_model(df: pd.DataFrame):
    feature_cols = [
        "xg_diff",
        "shots_diff",
        "prog_passes_diff",
        "prog_total_m_diff",
        "press_mid_diff",
        "press_fin_diff",
    ]

    X = df[feature_cols].copy()
    y = df["goal_diff"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Evaluation ===")
    print(f"Matches: {len(df)}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.3f}")
    print(f"R²: {r2_score(y_test, preds):.3f}")

    coefs = pd.Series(
        model.named_steps["ridge"].coef_, index=feature_cols
    ).sort_values(key=lambda s: s.abs(), ascending=False)

    print("\n=== Coefficients (importance) ===")
    print(coefs)

    # ---- SAVE artifacts for Streamlit app ----
    joblib.dump(model, "goal_diff_model.joblib")
    with open("goal_diff_feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print("\nSaved model: goal_diff_model.joblib")
    print("Saved feature cols: goal_diff_feature_cols.json")

    return model, feature_cols


# =========================
# Main
# =========================
if __name__ == "__main__":
    root_dir = "."
    df = build_dataset(root_dir)

    print("\n=== Dataset preview ===")
    print(df.head())
    print(f"\nTotal matches loaded: {len(df)}")

    df.to_csv("ml_features_dataset.csv", index=False)
    print("Saved ml_features_dataset.csv")

    print("\n=== Feature variability (nunique) ===")
    print(df[["prog_passes_diff", "prog_total_m_diff",
          "press_mid_diff", "press_fin_diff"]].nunique())

    if len(df) >= 20:
        train_model(df)
    else:
        print("Not training: need at least 20 matches.")
