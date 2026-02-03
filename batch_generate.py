from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import random
import time

import pandas as pd

# Reusamos tus funciones
from app import get_competitions, get_matches, get_events, match_context, save_df


# =========================
# Config
# =========================
OUTPUT_ROOT = Path(".")          # donde se crearán statsbomb_match_*
N_TARGET = 100                  # cuántos partidos quieres
RANDOM_SEED = 42
SAVE_CSV = True
SAVE_PARQUET = True

# Para no saturar el endpoint (conservador)
SLEEP_BETWEEN_MATCHES_SEC = 0.25
SLEEP_BETWEEN_LIST_CALLS_SEC = 0.10

# Si ya existe la carpeta del match, lo salta (para reanudar)
SKIP_IF_EXISTS = True


def _detect_cols_comps(comps: pd.DataFrame) -> Tuple[str, str]:
    cid_col = "competition_id" if "competition_id" in comps.columns else "competition"
    sid_col = "season_id" if "season_id" in comps.columns else "season"
    return cid_col, sid_col


def _detect_mid_col(matches: pd.DataFrame) -> str:
    if "match_id" in matches.columns:
        return "match_id"
    # fallback extra
    for c in matches.columns:
        if c.lower() == "match_id":
            return c
    raise ValueError("No se encontró columna match_id en el DF de matches.")


def build_match_index() -> pd.DataFrame:
    """
    Construye un índice global con columnas:
      competition_id, season_id, match_id
    recorriendo todas las filas de get_competitions() y llamando get_matches(cid,sid).
    """
    comps = get_competitions()
    if comps is None or comps.empty:
        raise RuntimeError("get_competitions() devolvió vacío.")

    cid_col, sid_col = _detect_cols_comps(comps)

    rows = []
    for _, r in comps.iterrows():
        try:
            cid = int(r[cid_col])
            sid = int(r[sid_col])

            matches = get_matches(cid, sid)
            time.sleep(SLEEP_BETWEEN_LIST_CALLS_SEC)

            if matches is None or matches.empty:
                continue

            mid_col = _detect_mid_col(matches)
            mids = matches[mid_col].dropna()

            for mid in mids:
                try:
                    rows.append(
                        {"competition_id": cid, "season_id": sid, "match_id": int(mid)})
                except Exception:
                    continue
        except Exception:
            continue

    if not rows:
        raise RuntimeError(
            "No se pudieron recolectar match_ids desde competiciones/temporadas.")

    idx = pd.DataFrame(rows).drop_duplicates(subset=["match_id"])
    return idx


def pick_random_matches(idx: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Toma n matches al azar del índice global.
    """
    if len(idx) <= n:
        return idx.sample(frac=1, random_state=seed).reset_index(drop=True)

    return idx.sample(n=n, random_state=seed).reset_index(drop=True)


def outdir_for_match(match_id: int) -> Path:
    outdir = OUTPUT_ROOT / f"statsbomb_match_{int(match_id)}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def generate_one(match_row: pd.Series) -> None:
    cid = int(match_row["competition_id"])
    sid = int(match_row["season_id"])
    mid = int(match_row["match_id"])

    outdir = outdir_for_match(mid)

    if SKIP_IF_EXISTS and (outdir / "events.csv").exists() and (outdir / "match.csv").exists():
        print(f"[SKIP] match_id={mid} (ya existe)")
        return

    # Descarga eventos del match
    events = get_events(mid)

    # Para guardar match.csv con score y nombres, volvemos a pedir matches de ese cid/sid
    matches_df = get_matches(cid, sid)
    mid_col = _detect_mid_col(matches_df)

    match_df = matches_df[matches_df[mid_col].astype(int) == mid].copy()
    if match_df.empty:
        # fallback: al menos guardar metadata mínima
        match_df = pd.DataFrame(
            [{"match_id": mid, "competition_id": cid, "season_id": sid}])

    # Contexto (para debug/log)
    try:
        ctx = match_context(matches_df, mid)
        ht = ctx.get("home_team") or ctx.get("home_team_name") or "Home"
        at = ctx.get("away_team") or ctx.get("away_team_name") or "Away"
    except Exception:
        ht, at = "Home", "Away"

    # Guardar usando tu función robusta (CSV + Parquet) :contentReference[oaicite:2]{index=2}
    save_df(events, outdir, "events", save_csv=SAVE_CSV,
            save_parquet=SAVE_PARQUET)
    save_df(match_df, outdir, "match",
            save_csv=SAVE_CSV, save_parquet=SAVE_PARQUET)

    print(
        f"[OK] match_id={mid} | cid={cid} sid={sid} | {ht} vs {at} | saved -> {outdir}")


def main():
    random.seed(RANDOM_SEED)

    print("Construyendo índice global de partidos (puede tardar según la API)…")
    idx = build_match_index()
    print(f"Total match_ids disponibles: {len(idx)}")

    picked = pick_random_matches(idx, N_TARGET, RANDOM_SEED)
    print(f"Seleccionados para generar: {len(picked)}")

    ok = 0
    fail = 0

    for i, row in picked.iterrows():
        mid = int(row["match_id"])
        print(f"\n[{i+1}/{len(picked)}] Generando match_id={mid} ...")
        try:
            generate_one(row)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] match_id={mid} -> {e}")
        time.sleep(SLEEP_BETWEEN_MATCHES_SEC)

    print(f"\nDONE ✅  ok={ok}  fail={fail}")
    print("Ahora corre: python3 ml_validate.py")


if __name__ == "__main__":
    main()
