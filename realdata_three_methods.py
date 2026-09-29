"""
Real-data comparison of the landmark Transformer, time-dependent Cox model,
and RDSM for the divorce data.

Keep this file, ``rdsm_model.py``, and ``Divorce_data.csv`` in the same folder.
Then run:
    python realdata_three_methods.py

The defaults below are the settings used for the final analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

import rdsm_model as rdsm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Variables used in the analysis
BASELINE_COVARIATES = [
    "H1AGE", "W1AGE",
    "H1ACE", "W1ACE",
    "H1PARENTMARR", "W1PARENTMARR",
    "H1ABUSE", "W1ABUSE",
    "H1EDUC", "W1EDUC",
    "T1_household_comp",
    "T1_Couple_unemployment",
    "T1_Neighbor_Employ",
    "T1_Neighbor_Income",

    "T1_Couple_Hostility",
    "T1_Income",
    "H1POS", "W1POS",
    "H1NEG", "W1NEG",
    "H1EFF", "W1EFF",
    "H1Substance", "W1Substance",
    "H1Neg_Affect", "W1Neg_Affect",
]

ASSESSMENT_YEARS = np.asarray(
    [0.00, 0.82, 1.57, 2.34, 4.15, 8.77, 9.94, 10.76, 11.44, 12.04],
    dtype=np.float64,
)
N_TIME = len(ASSESSMENT_YEARS)

TRANSFORMER_D_MODEL = 16
TRANSFORMER_NHEAD = 2
TRANSFORMER_FFN = 64
TRANSFORMER_DROPOUT = 0.30
TRANSFORMER_WEIGHT_DECAY = 1e-3
TRANSFORMER_LAYERS = 1
TRANSFORMER_MAX_EPOCHS = 10

COX_RIDGE = 1e-5

TV_COLUMN_MAP = {
    "W_rel_sat": {t: f"W{t}rel_sat" for t in range(1, 11)},
    "H_rel_sat": {t: f"H{t}rel_sat" for t in range(1, 11)},
    "W_stress": {t: f"W{t}stress" for t in range(1, 8)},
    "H_stress": {t: f"H{t}stress" for t in range(1, 8)},
}
COUPLE_TV_COLUMN_MAP = {
    "Couple_Fin_Strain": {
        t: (f"H{t}Fin_Strain", f"W{t}Fin_Strain") for t in range(1, 11)
    },
}
TV_NAMES = [
    "W_rel_sat", "H_rel_sat", "W_stress", "H_stress", "Couple_Fin_Strain"
]

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace("", np.nan), errors="coerce")

def _partner_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    out = np.full(a.shape, np.nan, dtype=float)
    both = fa & fb
    out[both] = (a[both] + b[both]) / 2.0
    out[fa & ~fb] = a[fa & ~fb]
    out[fb & ~fa] = b[fb & ~fa]
    return out

def load_divorce_wide(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)

    if "T1_Couple_Hostility" not in df.columns:
        if "H1Hostility" not in df.columns or "W1Hostility" not in df.columns:
            raise ValueError("Need H1Hostility and W1Hostility to construct T1_Couple_Hostility.")
        h1 = _to_numeric_series(df["H1Hostility"]).to_numpy(dtype=float)
        w1 = _to_numeric_series(df["W1Hostility"]).to_numpy(dtype=float)
        df["T1_Couple_Hostility"] = _partner_mean(h1, w1)

    required = ["csid", *BASELINE_COVARIATES]
    required += [col for wave_map in TV_COLUMN_MAP.values() for col in wave_map.values()]
    required += [
        col for wave_map in COUPLE_TV_COLUMN_MAP.values()
        for pair in wave_map.values() for col in pair
    ]
    required += [f"T{t}_STATUS" for t in range(1, 11)]
    missing_cols = sorted(set(c for c in required if c not in df.columns))
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    ids = df["csid"].astype(str).to_numpy()
    baseline = np.column_stack([
        _to_numeric_series(df[c]).to_numpy(dtype=float) for c in BASELINE_COVARIATES
    ])

    tv = np.full((len(df), N_TIME, len(TV_NAMES)), np.nan, dtype=float)
    name_to_j = {name: j for j, name in enumerate(TV_NAMES)}
    for name, wave_map in TV_COLUMN_MAP.items():
        j = name_to_j[name]
        for t, col in wave_map.items():
            tv[:, t - 1, j] = _to_numeric_series(df[col]).to_numpy(dtype=float)
    for name, wave_map in COUPLE_TV_COLUMN_MAP.items():
        j = name_to_j[name]
        for t, (h_col, w_col) in wave_map.items():
            h = _to_numeric_series(df[h_col]).to_numpy(dtype=float)
            w = _to_numeric_series(df[w_col]).to_numpy(dtype=float)
            tv[:, t - 1, j] = _partner_mean(h, w)

    status = np.column_stack([
        _to_numeric_series(df[f"T{t}_STATUS"]).to_numpy(dtype=float)
        for t in range(1, 11)
    ])

    observed_time = np.zeros(len(df), dtype=np.int64)
    observed_year = np.zeros(len(df), dtype=np.float64)
    event_observed = np.zeros(len(df), dtype=np.int64)
    for i in range(len(df)):
        event_idx = np.where(status[i] == 1)[0]
        if len(event_idx) > 0:
            idx = int(event_idx[0])
            observed_time[i] = idx + 1
            observed_year[i] = ASSESSMENT_YEARS[idx]
            event_observed[i] = 1
        else:
            observed_zero = np.where(status[i] == 0)[0]
            if len(observed_zero) == 0:
                raise ValueError(f"Person {ids[i]} has no observed status value.")
            idx = int(observed_zero[-1])
            observed_time[i] = idx + 1
            observed_year[i] = ASSESSMENT_YEARS[idx]
            event_observed[i] = 0

    visit_mask = np.any(np.isfinite(tv[:, :, :4]), axis=2).astype(np.float32)
    tv_missing_mask = (~np.isfinite(tv)).astype(np.float32)
    baseline_missing_mask = (~np.isfinite(baseline)).astype(np.float32)

    return {
        "ids": ids,
        "baseline": baseline,
        "baseline_missing_mask": baseline_missing_mask,
        "tv": tv,
        "tv_missing_mask": tv_missing_mask,
        "visit_mask": visit_mask,
        "status": status,
        "observed_time": observed_time,
        "observed_year": observed_year,
        "event_observed": event_observed,
        "n_original": len(df),
    }

def eligible_person_mask(data: Dict[str, np.ndarray], min_history_visits: int) -> np.ndarray:
    vm = data["visit_mask"]
    ot = data["observed_time"]
    keep = np.zeros(len(ot), dtype=bool)
    for i in range(len(ot)):
        keep[i] = int(vm[i, : max(int(ot[i]) - 1, 0)].sum()) >= int(min_history_visits)
    return keep

def subset_people(data: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.asarray(idx)
    n = len(data["ids"])
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out

@dataclass
class Preprocessor:
    baseline_mean: np.ndarray
    baseline_sd: np.ndarray
    tv_mean: np.ndarray
    tv_sd: np.ndarray

# Preprocessing. Means and SDs are estimated from the training sample only.
def fit_preprocessor(data: Dict[str, np.ndarray]) -> Preprocessor:
    b = np.asarray(data["baseline"], dtype=float)
    baseline_mean = np.nanmean(b, axis=0)
    baseline_mean = np.where(np.isfinite(baseline_mean), baseline_mean, 0.0)
    b_imp = np.where(np.isfinite(b), b, baseline_mean[None, :])
    baseline_sd = np.nanstd(b_imp, axis=0, ddof=0)
    baseline_sd = np.where((baseline_sd > 1e-8) & np.isfinite(baseline_sd), baseline_sd, 1.0)

    tv = np.asarray(data["tv"], dtype=float)
    tv_mean = np.nanmean(tv, axis=(0, 1))
    tv_mean = np.where(np.isfinite(tv_mean), tv_mean, 0.0)
    tv_sd = np.zeros(tv.shape[2], dtype=float)
    for j in range(tv.shape[2]):
        vals = tv[:, :, j]
        vals = vals[np.isfinite(vals)]
        s = np.std(vals, ddof=0) if len(vals) else 1.0
        tv_sd[j] = s if np.isfinite(s) and s > 1e-8 else 1.0

    return Preprocessor(
        baseline_mean=baseline_mean.astype(np.float64),
        baseline_sd=baseline_sd.astype(np.float64),
        tv_mean=tv_mean.astype(np.float64),
        tv_sd=tv_sd.astype(np.float64),
    )

def build_delta_time(visit_mask: np.ndarray) -> np.ndarray:
    vm = np.asarray(visit_mask, dtype=np.float32)
    N, T = vm.shape
    dt = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        last_visit_year = None
        for t in range(T):
            current_year = float(ASSESSMENT_YEARS[t])
            dt[i, t] = 0.0 if last_visit_year is None else current_year - last_visit_year
            if vm[i, t] > 0:
                last_visit_year = current_year
    return dt.astype(np.float32)

def transform_people(data: Dict[str, np.ndarray], pp: Preprocessor) -> Dict[str, np.ndarray]:
    n = len(data["ids"])
    T = data["tv"].shape[1]

    b = np.asarray(data["baseline"], dtype=float)
    bmiss = np.asarray(data["baseline_missing_mask"], dtype=np.float32)
    b_imp = np.where(np.isfinite(b), b, pp.baseline_mean[None, :])
    b_std = (b_imp - pp.baseline_mean[None, :]) / pp.baseline_sd[None, :]

    tv = np.asarray(data["tv"], dtype=float)
    tvmiss = np.asarray(data["tv_missing_mask"], dtype=np.float32)
    vm = np.asarray(data["visit_mask"], dtype=np.float32)

    tv_imp = np.empty_like(tv, dtype=np.float64)
    for i in range(n):
        last = pp.tv_mean.copy()
        for t in range(T):
            row = tv[i, t].copy()
            observed = np.isfinite(row)
            row[~observed] = last[~observed]
            tv_imp[i, t] = row
            last[observed] = row[observed]
    tv_std = (tv_imp - pp.tv_mean[None, None, :]) / pp.tv_sd[None, None, :]

    X = np.concatenate(
        [np.repeat(b_std[:, None, :], T, axis=1), tv_std], axis=2
    ).astype(np.float32)

    x_missing_mask = np.concatenate(
        [np.repeat(bmiss[:, None, :], T, axis=1), tvmiss], axis=2
    ).astype(np.float32)

    return {
        "ids": data["ids"],
        "X": X,
        "x_missing_mask": x_missing_mask,
        "visit_mask": vm.astype(np.float32),
        "delta_t": build_delta_time(vm),
        "observed_time": np.asarray(data["observed_time"], dtype=np.int64),
        "observed_year": np.asarray(data["observed_year"], dtype=np.float64),
        "event_observed": np.asarray(data["event_observed"], dtype=np.int64),
    }

# Landmark records used by all three models.
def build_multiple_landmark_samples(
    proc: Dict[str, np.ndarray], min_history_visits: int
) -> Dict[str, np.ndarray]:
    X = proc["X"]
    vm = proc["visit_mask"]
    xm = proc["x_missing_mask"]
    dt = proc["delta_t"]
    ot = proc["observed_time"]
    oy = proc["observed_year"]
    eo = proc["event_observed"]
    N, T, P = X.shape

    records = []
    for i in range(N):
        person_records = []
        for lm_idx in np.where(vm[i] > 0)[0]:
            s = int(lm_idx + 1)
            if s >= int(ot[i]):
                continue
            if int(vm[i, :s].sum()) < int(min_history_visits):
                continue
            residual_wave = int(ot[i] - s)
            residual_year = float(oy[i] - ASSESSMENT_YEARS[lm_idx])
            if residual_wave < 1 or residual_year <= 0:
                continue

            x_seq = np.zeros((T, P), dtype=np.float32)
            vm_seq = np.zeros(T, dtype=np.float32)
            xm_seq = np.zeros((T, P), dtype=np.float32)
            dt_seq = np.zeros(T, dtype=np.float32)
            x_seq[:s] = X[i, :s]
            vm_seq[:s] = vm[i, :s]
            xm_seq[:s] = xm[i, :s]
            dt_seq[:s] = dt[i, :s]
            person_records.append({
                "X": x_seq,
                "visit_mask": vm_seq,
                "x_missing_mask": xm_seq,
                "delta_t": dt_seq,
                "event_time": residual_wave,
                "event_time_years": residual_year,
                "event_observed": int(eo[i]),
                "person_idx": i,
                "person_id": str(proc["ids"][i]),
                "landmark_time": s,
                "landmark_year": float(ASSESSMENT_YEARS[lm_idx]),
            })

        m = len(person_records)
        for rec in person_records:
            rec["sample_weight"] = 1.0 / float(m)
            records.append(rec)

    if not records:
        raise ValueError("No eligible landmark records were created.")

    weights = np.asarray([r["sample_weight"] for r in records], dtype=np.float32)
    weights = weights / weights.mean()
    return {
        "X": np.stack([r["X"] for r in records]).astype(np.float32),
        "visit_mask": np.stack([r["visit_mask"] for r in records]).astype(np.float32),
        "x_missing_mask": np.stack([r["x_missing_mask"] for r in records]).astype(np.float32),
        "delta_t": np.stack([r["delta_t"] for r in records]).astype(np.float32),
        "event_time": np.asarray([r["event_time"] for r in records], dtype=np.int64),
        "event_time_years": np.asarray([r["event_time_years"] for r in records], dtype=np.float64),
        "event_observed": np.asarray([r["event_observed"] for r in records], dtype=np.int64),
        "sample_weight": weights,
        "person_idx": np.asarray([r["person_idx"] for r in records], dtype=np.int64),
        "person_id": np.asarray([r["person_id"] for r in records], dtype=object),
        "landmark_time": np.asarray([r["landmark_time"] for r in records], dtype=np.int64),
        "landmark_year": np.asarray([r["landmark_year"] for r in records], dtype=np.float64),
    }

# Evaluation helpers
def hazards_to_survival_and_cif(hazards: np.ndarray):
    h = np.clip(np.asarray(hazards, dtype=float), 1e-8, 1 - 1e-8)
    N, T = h.shape
    survival = np.ones((N, T), dtype=float)
    pmf = np.zeros((N, T), dtype=float)
    cif = np.zeros((N, T), dtype=float)
    prev = np.ones(N, dtype=float)
    for t in range(T):
        pmf[:, t] = prev * h[:, t]
        survival[:, t] = prev * (1 - h[:, t])
        cif[:, t] = 1 - survival[:, t]
        prev = survival[:, t]
    return survival, pmf, cif

def residual_cif_probability(hazards: np.ndarray, horizon: int) -> np.ndarray:
    _, _, cif = hazards_to_survival_and_cif(hazards)
    if horizon <= 0:
        return np.zeros(hazards.shape[0], dtype=float)
    if horizon > cif.shape[1]:
        return cif[:, -1]
    return cif[:, horizon - 1]

def make_equal_person_weights(person_id: np.ndarray) -> np.ndarray:
    p = np.asarray(person_id).astype(str)
    if len(p) == 0:
        return np.asarray([], dtype=float)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv].astype(float)

def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

def weighted_binary_auc(y, score, w):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    w = np.asarray(w, dtype=float)
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        return np.nan
    ps, ns = score[pos], score[neg]
    pw, nw = w[pos], w[neg]
    pair_w = pw[:, None] * nw[None, :]
    denom = pair_w.sum()
    if denom <= 0:
        return np.nan
    comp = (ps[:, None] > ns[None, :]).astype(float)
    ties = (ps[:, None] == ns[None, :]).astype(float)
    return float(np.sum(pair_w * (comp + 0.5 * ties)) / denom)

def weighted_risk_c_index(event_time, event_observed, risk, person_id, weights):
    t = np.asarray(event_time, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    r = np.asarray(risk, dtype=float)
    pid = np.asarray(person_id).astype(str)
    w = np.asarray(weights, dtype=float)
    concord = 0.0
    comparable = 0.0
    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            if pid[i] == pid[j]:
                continue
            pair_w = w[i] * w[j]
            if e[i] == 1 and t[i] < t[j]:
                comparable += pair_w
                concord += pair_w * (1.0 if r[i] > r[j] else 0.5 if r[i] == r[j] else 0.0)
            elif e[j] == 1 and t[j] < t[i]:
                comparable += pair_w
                concord += pair_w * (1.0 if r[j] > r[i] else 0.5 if r[i] == r[j] else 0.0)
    return np.nan if comparable <= 0 else float(concord / comparable)

def weighted_ece(y, p, w, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total <= 0:
        return np.nan
    edges = np.linspace(0, 1, n_bins + 1)
    ans = 0.0
    for b in range(n_bins):
        if b < n_bins - 1:
            m = (p >= edges[b]) & (p < edges[b + 1])
        else:
            m = (p >= edges[b]) & (p <= edges[b + 1])
        if not np.any(m):
            continue
        bw = w[m].sum()
        obs = np.sum(w[m] * y[m]) / bw
        pred = np.sum(w[m] * p[m]) / bw
        ans += (bw / total) * abs(obs - pred)
    return float(ans)

def most_likely_residual_event_time_within_study(
    hazards: np.ndarray,
    landmark_time: np.ndarray,
    n_time: int = N_TIME,
) -> Tuple[np.ndarray, np.ndarray]:
    _, pmf, _ = hazards_to_survival_and_cif(hazards)
    landmark_time = np.asarray(landmark_time, dtype=np.int64)
    out_wave = np.full(len(landmark_time), np.nan, dtype=float)
    out_year = np.full(len(landmark_time), np.nan, dtype=float)
    for i in range(len(landmark_time)):
        s = int(landmark_time[i])
        max_r = min(pmf.shape[1], int(n_time) - s)
        if max_r <= 0:
            continue
        r = int(np.argmax(pmf[i, :max_r]) + 1)
        out_wave[i] = float(r)
        out_year[i] = float(ASSESSMENT_YEARS[s + r - 1] - ASSESSMENT_YEARS[s - 1])
    return out_wave, out_year

def weighted_event_time_c_index(event_time, event_observed, predicted_time, person_id, weights):
    t = np.asarray(event_time, dtype=float)
    e = np.asarray(event_observed, dtype=int)
    p = np.asarray(predicted_time, dtype=float)
    pid = np.asarray(person_id).astype(str)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(t) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    t, e, p, pid, w = t[mask], e[mask], p[mask], pid[mask], w[mask]
    concord = 0.0
    comparable = 0.0
    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            if pid[i] == pid[j]:
                continue
            pair_w = w[i] * w[j]
            if pair_w <= 0:
                continue
            if e[i] == 1 and t[i] < t[j]:
                comparable += pair_w
                concord += pair_w * (1.0 if p[i] < p[j] else 0.5 if p[i] == p[j] else 0.0)
            elif e[j] == 1 and t[j] < t[i]:
                comparable += pair_w
                concord += pair_w * (1.0 if p[j] < p[i] else 0.5 if p[i] == p[j] else 0.0)
    return np.nan if comparable <= 0 else float(concord / comparable)

def evaluate_event_time_predictions(
    predicted_residual_time_years: np.ndarray,
    lm: Dict[str, np.ndarray],
) -> Dict[str, float]:
    pred = np.asarray(predicted_residual_time_years, dtype=float)
    obs_year = np.asarray(lm["event_time_years"], dtype=float)
    valid = np.isfinite(pred) & np.isfinite(obs_year)
    pid_all = np.asarray(lm["person_id"]).astype(str)

    if np.any(valid):
        c_weights = make_equal_person_weights(pid_all[valid])
        c_index = weighted_event_time_c_index(
            obs_year[valid],
            lm["event_observed"][valid],
            pred[valid],
            pid_all[valid],
            c_weights,
        )
    else:
        c_index = np.nan

    event_mask = valid & (np.asarray(lm["event_observed"], dtype=int) == 1)
    if np.any(event_mask):
        err = np.abs(pred[event_mask] - obs_year[event_mask])
        event_weights = make_equal_person_weights(pid_all[event_mask])
        mae = weighted_mean(err, event_weights)
    else:
        mae = np.nan

    return {
        "MAE_event_time": mae,
        "EventTime_C-index": c_index,
        "n_event_time_eval": int(event_mask.sum()),
        "n_person_event_time_eval": int(len(np.unique(pid_all[event_mask]))) if np.any(event_mask) else 0,
    }

def evaluate_hazards(
    hazards: np.ndarray,
    lm: Dict[str, np.ndarray],
    horizons: Sequence[int],
    n_time: int = N_TIME,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    horizon_rows = []
    metrics = {}
    for a in horizons:
        prob = residual_cif_probability(hazards, int(a))
        structural = (lm["landmark_time"] + int(a)) <= int(n_time)
        known = ((lm["event_observed"] == 1) & (lm["event_time"] <= int(a))) | (lm["event_time"] >= int(a))
        valid = structural & known
        y = ((lm["event_observed"] == 1) & (lm["event_time"] <= int(a)))[valid].astype(int)
        p = prob[valid]
        pid = lm["person_id"][valid]
        w = make_equal_person_weights(pid)

        auc = weighted_binary_auc(y, p, w)
        brier = weighted_mean((p - y) ** 2, w)
        pclip = np.clip(p, 1e-8, 1 - 1e-8)
        nll = weighted_mean(-(y * np.log(pclip) + (1 - y) * np.log(1 - pclip)), w)
        ece = weighted_ece(y, p, w)
        rc = weighted_risk_c_index(
            lm["event_time_years"][valid], lm["event_observed"][valid], p, pid, w
        )
        metrics.update({
            f"AUC@{a}": auc,
            f"Risk_C-index@{a}": rc,
            f"Brier@{a}": brier,
            f"Binary_NLL@{a}": nll,
            f"ECE@{a}": ece,
            f"n_eval@{a}": int(valid.sum()),
            f"n_person_eval@{a}": int(len(np.unique(pid))),
        })

        for idx_global, yy, pp in zip(np.where(valid)[0], y, p):
            horizon_rows.append({
                "record_index": int(idx_global),
                "person_id": str(lm["person_id"][idx_global]),
                "landmark_time": int(lm["landmark_time"][idx_global]),
                "landmark_year": float(lm["landmark_year"][idx_global]),
                "residual_observed_waves": int(lm["event_time"][idx_global]),
                "residual_observed_years": float(lm["event_time_years"][idx_global]),
                "event_observed": int(lm["event_observed"][idx_global]),
                "horizon": int(a),
                "y_true": int(yy),
                "predicted_risk": float(pp),
            })

    pred_event_wave, pred_event_year = most_likely_residual_event_time_within_study(
        hazards, lm["landmark_time"], n_time=n_time
    )
    metrics.update(evaluate_event_time_predictions(pred_event_year, lm))
    event_rows = pd.DataFrame({
        "record_index": np.arange(len(lm["event_time"]), dtype=int),
        "person_id": np.asarray(lm["person_id"]).astype(str),
        "landmark_time": np.asarray(lm["landmark_time"], dtype=int),
        "landmark_year": np.asarray(lm["landmark_year"], dtype=float),
        "residual_observed_waves": np.asarray(lm["event_time"], dtype=int),
        "residual_observed_years": np.asarray(lm["event_time_years"], dtype=float),
        "event_observed": np.asarray(lm["event_observed"], dtype=int),
        "predicted_residual_event_waves": pred_event_wave,
        "predicted_residual_event_years": pred_event_year,
    })
    return metrics, pd.DataFrame(horizon_rows), event_rows

def metrics_long(method: str, fold: int, metrics: Dict[str, float]) -> List[dict]:
    rows = []
    for k, v in metrics.items():
        if k.startswith("n_"):
            continue
        rows.append({"fold": fold, "method": method, "metric": k, "value": v})
    return rows

# Transformer
class LandmarkDataset(Dataset):
    def __init__(self, lm: Dict[str, np.ndarray]):
        self.X = torch.tensor(lm["X"], dtype=torch.float32)
        self.vm = torch.tensor(lm["visit_mask"], dtype=torch.float32)
        self.xm = torch.tensor(lm["x_missing_mask"], dtype=torch.float32)
        self.dt = torch.tensor(lm["delta_t"], dtype=torch.float32)
        self.t = torch.tensor(lm["event_time"], dtype=torch.long)
        self.e = torch.tensor(lm["event_observed"], dtype=torch.float32)
        self.w = torch.tensor(lm["sample_weight"], dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return {"X": self.X[i], "vm": self.vm[i], "xm": self.xm[i], "dt": self.dt[i], "t": self.t[i], "e": self.e[i], "w": self.w[i]}

class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        return self.mlp(x)

class SurvivalTransformer(nn.Module):
    def __init__(self, n_features, n_time, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, dropout=TRANSFORMER_DROPOUT):
        super().__init__()
        input_dim = 2 * n_features + 2
        self.x_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.time_enc = TimeEncoding(d_model)
        self.register_buffer(
            "assessment_years",
            torch.tensor(ASSESSMENT_YEARS[:n_time], dtype=torch.float32),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=TRANSFORMER_FFN, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.out = nn.Linear(d_model, n_time)

    def forward(self, X, vm, xm, dt):
        B, T, _ = X.shape
        inp = torch.cat([X, xm, vm.unsqueeze(-1), dt.unsqueeze(-1)], dim=-1)
        h = self.x_proj(inp)
        tg = self.assessment_years[:T].view(1, T, 1)
        h = h + self.time_enc(tg)
        padding = vm == 0
        causal = torch.triu(torch.ones(T, T, device=X.device, dtype=torch.bool), diagonal=1)
        z = self.encoder(h, mask=causal, src_key_padding_mask=padding)
        ti = torch.arange(T, device=X.device).view(1, T).expand(B, T)
        last = ti.masked_fill(vm <= 0, -1).max(dim=1).values.clamp(min=0)
        pooled = z[torch.arange(B, device=X.device), last]
        return self.out(pooled)

def discrete_nll(logits, event_time, event_observed, weights, event_weight=1.0):
    h = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    T = h.shape[1]
    idx = torch.arange(1, T + 1, device=h.device).view(1, -1)
    t = event_time.view(-1, 1)
    before = (idx < t).float()
    through = (idx <= t).float()
    at = (idx == t).float()
    log_surv = torch.log(1 - h)
    log_h = torch.log(h)
    l_event = -(before * log_surv).sum(1) - (at * log_h).sum(1)
    l_cens = -(through * log_surv).sum(1)
    loss = event_observed * event_weight * l_event + (1 - event_observed) * l_cens
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)

def make_loader(lm, batch_size, shuffle):
    return DataLoader(LandmarkDataset(lm), batch_size=batch_size, shuffle=shuffle)

def run_transformer_epoch(model, loader, optimizer, device, event_weight, train):
    model.train(train)
    total = 0.0
    n = 0
    for b in loader:
        X, vm, xm, dt = b["X"].to(device), b["vm"].to(device), b["xm"].to(device), b["dt"].to(device)
        t, e, w = b["t"].to(device), b["e"].to(device), b["w"].to(device)
        logits = model(X, vm, xm, dt)
        loss = discrete_nll(logits, t, e, w, event_weight=event_weight)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach().cpu()) * len(t)
        n += len(t)
    return total / max(n, 1)

@torch.no_grad()
def transformer_predict(model, lm, batch_size, device):
    loader = make_loader(lm, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    for b in loader:
        logits = model(b["X"].to(device), b["vm"].to(device), b["xm"].to(device), b["dt"].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)

def train_transformer_stage1(train_lm, val_lm, seed, event_weight, epochs, lr, batch_size, horizons):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurvivalTransformer(train_lm["X"].shape[-1], N_TIME).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRANSFORMER_WEIGHT_DECAY)
    tr_loader = make_loader(train_lm, batch_size, True)
    va_loader = make_loader(val_lm, batch_size, False)
    best_state = None
    best_epoch = 1
    best_val = float("inf")
    bad = 0
    patience = 12

    for epoch in range(1, epochs + 1):
        run_transformer_epoch(model, tr_loader, opt, device, event_weight, True)
        val_loss = run_transformer_epoch(model, va_loader, None, device, event_weight, False)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hz = transformer_predict(model, val_lm, batch_size, device)
    val_metrics, _, _ = evaluate_hazards(hz, val_lm, horizons)
    return {"best_epoch": best_epoch, "best_val_loss": best_val, "metrics": val_metrics}

def choose_transformer_config(train_lm, val_lm, seed, w_grid, epochs, lr, batch_size, horizons):
    candidates = []
    higher_metrics = [f"Risk_C-index@{h}" for h in horizons]
    lower_metrics = (
        [f"Brier@{h}" for h in horizons]
        + [f"Binary_NLL@{h}" for h in horizons]
        + [f"ECE@{h}" for h in horizons]
    )
    selection_metrics = higher_metrics + lower_metrics

    for w in w_grid:
        result = train_transformer_stage1(
            train_lm, val_lm, seed, float(w), epochs, lr, batch_size, horizons
        )
        candidates.append({"w": float(w), "result": result, "ranks": {}})

    for metric in selection_metrics:
        vals = np.asarray(
            [c["result"]["metrics"].get(metric, np.nan) for c in candidates], dtype=float
        )
        finite = np.isfinite(vals)
        ranks = np.full(len(candidates), len(candidates) + 1.0)
        if np.any(finite):
            order_vals = -vals[finite] if metric in higher_metrics else vals[finite]
            order = np.argsort(order_vals, kind="mergesort")
            inds = np.where(finite)[0][order]
            for rank, idx in enumerate(inds, 1):
                ranks[idx] = rank
        for i, c in enumerate(candidates):
            c["ranks"][metric] = float(ranks[i])

    for c in candidates:
        c["mean_rank"] = float(np.mean([c["ranks"][m] for m in selection_metrics]))

    best = min(
        candidates,
        key=lambda c: (c["mean_rank"], c["result"]["best_val_loss"], c["w"]),
    )
    best["selection_metrics"] = selection_metrics
    best["candidate_details"] = [
        {
            "beta": c["w"],
            "mean_validation_rank": c["mean_rank"],
            "best_epoch": c["result"]["best_epoch"],
            "best_validation_loss": c["result"]["best_val_loss"],
            **{f"value_{m}": c["result"]["metrics"].get(m, np.nan) for m in selection_metrics},
            **{f"rank_{m}": c["ranks"].get(m, np.nan) for m in selection_metrics},
        }
        for c in candidates
    ]
    return best

def refit_transformer(dev_lm, seed, event_weight, epochs, lr, batch_size):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurvivalTransformer(dev_lm["X"].shape[-1], N_TIME).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRANSFORMER_WEIGHT_DECAY)
    loader = make_loader(dev_lm, batch_size, True)
    for _ in range(max(1, int(epochs))):
        run_transformer_epoch(model, loader, opt, device, float(event_weight), True)
    return model, device

# Time-dependent Cox model
@dataclass
class TimeDependentCoxModel:
    beta: np.ndarray
    event_times: np.ndarray
    cum_baseline_hazard: np.ndarray
    ridge: float
    converged: bool
    n_iter: int
    final_loglik: float

    def baseline_cumhaz_at(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.event_times, t, side="right") - 1
        out = np.zeros_like(t, dtype=float)
        m = idx >= 0
        out[m] = self.cum_baseline_hazard[idx[m]]
        return out

    def survival(self, absolute_time, X):

        H0 = self.baseline_cumhaz_at(absolute_time)
        lp = np.asarray(X, dtype=float) @ self.beta
        return np.exp(-H0 * np.exp(np.clip(lp, -40, 40)))

def build_time_dependent_cox_rows(proc: Dict[str, np.ndarray]):
    X = np.asarray(proc["X"], dtype=float)
    observed_wave = np.asarray(proc["observed_time"], dtype=int)
    event_observed = np.asarray(proc["event_observed"], dtype=int)

    rows_x = []
    rows_start = []
    rows_stop = []
    rows_event = []

    for i in range(X.shape[0]):
        obs_wave = int(observed_wave[i])
        if obs_wave <= 1:
            continue

        for end_wave in range(2, obs_wave + 1):
            start_idx = end_wave - 2
            stop_idx = end_wave - 1
            rows_x.append(X[i, stop_idx, :])
            rows_start.append(float(ASSESSMENT_YEARS[start_idx]))
            rows_stop.append(float(ASSESSMENT_YEARS[stop_idx]))
            rows_event.append(
                1 if (event_observed[i] == 1 and end_wave == obs_wave) else 0
            )

    if not rows_x:
        raise ValueError("No time-dependent Cox training rows were created.")

    return (
        np.asarray(rows_x, dtype=np.float64),
        np.asarray(rows_start, dtype=np.float64),
        np.asarray(rows_stop, dtype=np.float64),
        np.asarray(rows_event, dtype=np.int64),
    )

def _cox_penalized_loglik(X, start, stop, event, beta, ridge):
    lp = X @ beta
    exp_lp = np.exp(np.clip(lp, -50.0, 50.0))
    unique_event_times = np.sort(np.unique(stop[event == 1]))
    out = 0.0
    for t in unique_event_times:
        risk = (start < t) & (stop >= t)
        died = (stop == t) & (event == 1)
        d = int(died.sum())
        if d == 0:
            continue
        denom = float(exp_lp[risk].sum())
        out += float(lp[died].sum()) - d * np.log(max(denom, 1e-300))
    out -= 0.5 * float(ridge) * float(beta @ beta)
    return float(out)

def fit_time_dependent_cox(
    proc: Dict[str, np.ndarray],
    ridge: float = COX_RIDGE,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> TimeDependentCoxModel:
    X, start, stop, event = build_time_dependent_cox_rows(proc)
    p = X.shape[1]
    beta = np.zeros(p, dtype=np.float64)
    unique_event_times = np.sort(np.unique(stop[event == 1]))

    if unique_event_times.size == 0:
        return TimeDependentCoxModel(
            beta=beta,
            event_times=np.asarray([], dtype=float),
            cum_baseline_hazard=np.asarray([], dtype=float),
            ridge=float(ridge),
            converged=True,
            n_iter=0,
            final_loglik=np.nan,
        )

    converged = False
    prev_obj = -np.inf
    final_obj = _cox_penalized_loglik(X, start, stop, event, beta, ridge)
    n_iter_run = 0

    for it in range(1, int(max_iter) + 1):
        n_iter_run = it
        lp = X @ beta
        exp_lp = np.exp(np.clip(lp, -50.0, 50.0))
        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)

        for t in unique_event_times:
            risk = (start < t) & (stop >= t)
            died = (stop == t) & (event == 1)
            d = int(died.sum())
            if d == 0:
                continue

            Xr = X[risk]
            wr = exp_lp[risk]
            denom = float(wr.sum())
            if denom <= 0:
                continue

            mean_x = (wr[:, None] * Xr).sum(axis=0) / denom
            second = Xr.T @ (Xr * wr[:, None]) / denom
            info_t = second - np.outer(mean_x, mean_x)

            grad += X[died].sum(axis=0) - d * mean_x
            hess -= d * info_t

        grad_pen = grad - float(ridge) * beta
        hess_pen = hess - float(ridge) * np.eye(p)
        info = -hess_pen

        try:
            step = np.linalg.solve(info, grad_pen)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ grad_pen

        proposal = beta + step
        proposal_obj = _cox_penalized_loglik(X, start, stop, event, proposal, ridge)

        n_halve = 0
        current_obj = _cox_penalized_loglik(X, start, stop, event, beta, ridge)
        while proposal_obj < current_obj and n_halve < 20:
            step *= 0.5
            proposal = beta + step
            proposal_obj = _cox_penalized_loglik(X, start, stop, event, proposal, ridge)
            n_halve += 1

        beta = proposal
        final_obj = proposal_obj

        if np.max(np.abs(step)) < tol or abs(proposal_obj - prev_obj) < tol:
            converged = True
            break
        prev_obj = proposal_obj

    lp = X @ beta
    exp_lp = np.exp(np.clip(lp, -50.0, 50.0))
    increments = []
    for t in unique_event_times:
        risk = (start < t) & (stop >= t)
        died = (stop == t) & (event == 1)
        d = float(died.sum())
        denom = float(exp_lp[risk].sum())
        increments.append(d / max(denom, 1e-300))

    return TimeDependentCoxModel(
        beta=beta,
        event_times=unique_event_times.astype(float),
        cum_baseline_hazard=np.cumsum(np.asarray(increments, dtype=float)),
        ridge=float(ridge),
        converged=bool(converged),
        n_iter=int(n_iter_run),
        final_loglik=float(final_obj),
    )

def cox_predict_hazards(
    model: TimeDependentCoxModel,
    lm: Dict[str, np.ndarray],
) -> np.ndarray:
    s_wave = np.asarray(lm["landmark_time"], dtype=int)
    X_landmark = lm["X"][np.arange(len(s_wave)), s_wave - 1, :].astype(float)
    n = len(s_wave)
    hz = np.full((n, N_TIME), 1e-8, dtype=float)

    for i in range(n):
        s = int(s_wave[i])
        max_r = N_TIME - s
        if max_r <= 0:
            continue
        for r in range(1, max_r + 1):
            prev_abs = float(ASSESSMENT_YEARS[s + r - 2])
            curr_abs = float(ASSESSMENT_YEARS[s + r - 1])
            s_prev = float(model.survival(np.asarray([prev_abs]), X_landmark[i:i+1])[0])
            s_curr = float(model.survival(np.asarray([curr_abs]), X_landmark[i:i+1])[0])
            hz[i, r - 1] = np.clip(
                1.0 - s_curr / max(s_prev, 1e-12),
                1e-8,
                1.0 - 1e-8,
            )
    return hz

# RDSM
def rdsm_person_sequences(proc: Dict[str, np.ndarray], min_history_visits: int):
    return rdsm.build_person_level_rdsm_sequences(
        X_imp=proc["X"],
        visit_mask=proc["visit_mask"],
        x_missing_mask=proc["x_missing_mask"],
        delta_t=proc["delta_t"],
        observed_time=proc["observed_time"],
        observed_year=proc["observed_year"],
        event_observed=proc["event_observed"],
        assessment_years=ASSESSMENT_YEARS,
        min_history_visits=min_history_visits,
    )

def split_inner_indices(event_observed, seed, val_fraction):
    y = np.asarray(event_observed, dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
    tr, va = next(splitter.split(np.zeros(len(y)), y))
    return tr, va

# One outer cross-validation fold
def run_fold(
    eligible_data: Dict[str, np.ndarray],
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    fold: int,
    seed: int,
    args,
):
    outer_train_raw = subset_people(eligible_data, outer_train_idx)
    outer_test_raw = subset_people(eligible_data, outer_test_idx)
    inner_tr_rel, inner_va_rel = split_inner_indices(outer_train_raw["event_observed"], seed + fold * 101, args.inner_val_fraction)
    inner_train_raw = subset_people(outer_train_raw, inner_tr_rel)
    inner_val_raw = subset_people(outer_train_raw, inner_va_rel)

    pp1 = fit_preprocessor(inner_train_raw)
    inner_train_proc = transform_people(inner_train_raw, pp1)
    inner_val_proc = transform_people(inner_val_raw, pp1)
    inner_train_lm = build_multiple_landmark_samples(inner_train_proc, args.min_history_visits)
    inner_val_lm = build_multiple_landmark_samples(inner_val_proc, args.min_history_visits)

    best_t = choose_transformer_config(
        inner_train_lm, inner_val_lm, seed + fold,
        args.w_grid, args.epochs, args.lr, args.batch_size, args.horizons,
    )

    xtr, ttr, etr, info_tr = rdsm_person_sequences(inner_train_proc, args.min_history_visits)
    xva, tva, eva, info_va = rdsm_person_sequences(inner_val_proc, args.min_history_visits)
    _, best_rdsm_iter, best_rdsm_val = rdsm.fit_rdsm_with_external_validation(
        x_train_streams=xtr, t_train_streams=ttr, e_train_streams=etr,
        x_val_streams=xva, t_val_streams=tva, e_val_streams=eva,
        seed=seed + fold,
        k=args.rdsm_k, hidden=args.rdsm_hidden, layers=args.rdsm_layers,
        typ=args.rdsm_type, distribution=args.rdsm_distribution,
        iters=args.rdsm_iters, learning_rate=args.rdsm_lr, batch_size=args.rdsm_batch_size,
    )

    pp_final = fit_preprocessor(outer_train_raw)
    dev_proc = transform_people(outer_train_raw, pp_final)
    test_proc = transform_people(outer_test_raw, pp_final)
    dev_lm = build_multiple_landmark_samples(dev_proc, args.min_history_visits)
    test_lm = build_multiple_landmark_samples(test_proc, args.min_history_visits)

    transformer, device = refit_transformer(
        dev_lm, seed + fold, best_t["w"], best_t["result"]["best_epoch"], args.lr, args.batch_size
    )
    t_hz = transformer_predict(transformer, test_lm, args.batch_size, device)

    cox = fit_time_dependent_cox(dev_proc, ridge=COX_RIDGE)
    c_hz = cox_predict_hazards(cox, test_lm)

    xdev, tdev, edev, info_dev = rdsm_person_sequences(dev_proc, args.min_history_visits)
    rdsm_model = rdsm.refit_rdsm_on_full_development(
        x_dev_streams=xdev, t_dev_streams=tdev, e_dev_streams=edev,
        selected_iters=best_rdsm_iter, seed=seed + fold,
        k=args.rdsm_k, hidden=args.rdsm_hidden, layers=args.rdsm_layers,
        typ=args.rdsm_type, distribution=args.rdsm_distribution,
        learning_rate=args.rdsm_lr, batch_size=args.rdsm_batch_size,
    )
    r_hz = rdsm.predict_landmark_residual_hazards(
        rdsm_model, test_lm, assessment_years=ASSESSMENT_YEARS
    )

    methods = {
        "Landmark Transformer": t_hz,
        "Time-dependent Cox: landmark-aligned": c_hz,
        "RDSM": r_hz,
    }

    fold_metric_rows = []
    horizon_pred_rows = []
    event_pred_rows = []
    for method, hz in methods.items():
        m, horizon_pred, event_pred = evaluate_hazards(hz, test_lm, args.horizons)
        fold_metric_rows.extend(metrics_long(method, fold, m))
        horizon_pred["method"] = method
        horizon_pred["fold"] = fold
        event_pred["method"] = method
        event_pred["fold"] = fold
        horizon_pred_rows.append(horizon_pred)
        event_pred_rows.append(event_pred)

    beta_details = []
    for row in best_t.get("candidate_details", []):
        beta_details.append({"fold": fold, **row, "selected": int(float(row["beta"]) == float(best_t["w"]))})

    hyper = {
        "fold": fold,
        "transformer_beta": best_t["w"],
        "transformer_event_weight": best_t["w"],
        "transformer_epoch": best_t["result"]["best_epoch"],
        "transformer_mean_validation_rank": best_t["mean_rank"],
        "transformer_beta_selection_metrics": ";".join(best_t.get("selection_metrics", [])),
        "cox_ridge": COX_RIDGE,
        "cox_ridge_tuned": 0,
        "cox_specification": "simulation-aligned start-stop time-dependent Cox",
        "cox_converged": int(cox.converged),
        "cox_n_iter": int(cox.n_iter),
        "cox_final_penalized_loglik": float(cox.final_loglik),
        "rdsm_selected_iter": int(best_rdsm_iter),
        "rdsm_best_validation_nll": float(best_rdsm_val),
        "n_outer_train_people": len(outer_train_idx),
        "n_outer_test_people": len(outer_test_idx),
        "n_test_landmarks": len(test_lm["event_time"]),
        "n_test_events_people": int(outer_test_raw["event_observed"].sum()),
    }
    return (
        fold_metric_rows,
        pd.concat(horizon_pred_rows, ignore_index=True),
        pd.concat(event_pred_rows, ignore_index=True),
        hyper,
        beta_details,
    )

# Summaries across folds
def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    g = fold_metrics.groupby(["method", "metric"], sort=False)["value"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"std": "sd", "count": "n_folds"})
    out["se"] = out["sd"] / np.sqrt(out["n_folds"].clip(lower=1))
    return out[["method", "metric", "mean", "sd", "se", "n_folds"]]

def pooled_horizon_metrics_from_predictions(pred: pd.DataFrame) -> List[dict]:
    rows = []
    for (method, horizon), g in pred.groupby(["method", "horizon"], sort=False):
        y = g["y_true"].to_numpy(dtype=int)
        p = g["predicted_risk"].to_numpy(dtype=float)
        pid = g["person_id"].astype(str).to_numpy()
        w = make_equal_person_weights(pid)
        auc = weighted_binary_auc(y, p, w)
        brier = weighted_mean((p - y) ** 2, w)
        pc = np.clip(p, 1e-8, 1 - 1e-8)
        nll = weighted_mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), w)
        ece = weighted_ece(y, p, w)
        rc = weighted_risk_c_index(
            g["residual_observed_years"].to_numpy(),
            g["event_observed"].to_numpy(), p, pid, w,
        )
        vals = {
            f"AUC@{horizon}": auc,
            f"Risk_C-index@{horizon}": rc,
            f"Brier@{horizon}": brier,
            f"Binary_NLL@{horizon}": nll,
            f"ECE@{horizon}": ece,
        }
        for metric, value in vals.items():
            rows.append({
                "method": method,
                "metric": metric,
                "value": value,
                "n_landmark_records": len(g),
                "n_people": g["person_id"].nunique(),
            })
    return rows

def pooled_event_time_metrics_from_predictions(pred: pd.DataFrame) -> List[dict]:
    rows = []
    for method, g in pred.groupby("method", sort=False):
        pid = g["person_id"].astype(str).to_numpy()
        pred_t = g["predicted_residual_event_years"].to_numpy(dtype=float)
        obs_t = g["residual_observed_years"].to_numpy(dtype=float)
        event = g["event_observed"].to_numpy(dtype=int)
        valid = np.isfinite(pred_t)
        c_weights = make_equal_person_weights(pid[valid]) if np.any(valid) else np.asarray([])
        cidx = weighted_event_time_c_index(
            obs_t[valid], event[valid], pred_t[valid], pid[valid], c_weights
        ) if np.any(valid) else np.nan
        emask = valid & (event == 1)
        if np.any(emask):
            eweights = make_equal_person_weights(pid[emask])
            mae = weighted_mean(np.abs(pred_t[emask] - obs_t[emask]), eweights)
        else:
            mae = np.nan
        rows.extend([
            {
                "method": method, "metric": "MAE_event_time", "value": mae,
                "n_landmark_records": int(emask.sum()),
                "n_people": int(len(np.unique(pid[emask]))) if np.any(emask) else 0,
            },
            {
                "method": method, "metric": "EventTime_C-index", "value": cidx,
                "n_landmark_records": int(valid.sum()),
                "n_people": int(len(np.unique(pid[valid]))) if np.any(valid) else 0,
            },
        ])
    return rows

def pooled_metrics_from_predictions(horizon_pred: pd.DataFrame, event_pred: pd.DataFrame) -> pd.DataFrame:
    rows = pooled_horizon_metrics_from_predictions(horizon_pred)
    rows.extend(pooled_event_time_metrics_from_predictions(event_pred))
    return pd.DataFrame(rows)

FINAL_METRIC_ORDER = [
    "AUC@3", "AUC@5",
    "Risk_C-index@3", "Risk_C-index@5",
    "Brier@3", "Brier@5",
    "Binary_NLL@3", "Binary_NLL@5",
    "ECE@3", "ECE@5",
    "MAE_event_time", "EventTime_C-index",
]

def _order_metric_rows(wide: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(FINAL_METRIC_ORDER)}
    out = wide.copy()
    out["_metric_order"] = out["metric"].map(rank).fillna(len(rank))
    out = out.sort_values(["_metric_order", "metric"], kind="stable").drop(columns="_metric_order")
    return out.reset_index(drop=True)

def make_wide_from_fold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="metric", columns="method", values="mean").reset_index()
    wide.columns.name = None
    return _order_metric_rows(wide)

def make_wide_from_pooled(pooled: pd.DataFrame) -> pd.DataFrame:
    wide = pooled.pivot(index="metric", columns="method", values="value").reset_index()
    wide.columns.name = None
    return _order_metric_rows(wide)

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

# Run the analysis
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).with_name("Divorce_data.csv")), help="Path to the analysis data")
    p.add_argument("--output-dir", default="results_divorce_three_methods")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-val-fraction", type=float, default=0.15)
    p.add_argument("--min-history-visits", type=int, default=3)
    p.add_argument("--horizons", type=str, default="3,5")
    p.add_argument("--audit-only", action="store_true", help="Validate data/preprocessing and write audit files, then stop.")

    p.add_argument("--epochs", type=int, default=TRANSFORMER_MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-grid", type=str, default="1,2,3,4,5,6,7,8,9,10")

    p.add_argument("--rdsm-k", type=int, default=3)
    p.add_argument("--rdsm-hidden", type=int, default=32)
    p.add_argument("--rdsm-layers", type=int, default=1)
    p.add_argument("--rdsm-type", default="LSTM")
    p.add_argument("--rdsm-distribution", default="Weibull")
    p.add_argument("--rdsm-iters", type=int, default=100)
    p.add_argument("--rdsm-lr", type=float, default=1e-3)
    p.add_argument("--rdsm-batch-size", type=int, default=64)
    args = p.parse_args()

    args.horizons = parse_int_list(args.horizons)
    args.w_grid = parse_float_list(args.w_grid)
    if any(h <= 0 or h >= N_TIME for h in args.horizons):
        raise ValueError("Horizons must be positive residual-wave integers smaller than 10.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_divorce_wide(args.data)
    keep = eligible_person_mask(raw, args.min_history_visits)
    eligible = subset_people(raw, np.where(keep)[0])

    audit = pd.DataFrame([
        {"item": "Analysis condition", "value": "S-FS 5-TV predictors + actual irregular assessment times + R1 Transformer + simulation-aligned Cox"},
        {"item": "Original participants", "value": raw["n_original"]},
        {"item": "Eligible participants", "value": len(eligible["ids"])},
        {"item": "Eligible observed events", "value": int(eligible["event_observed"].sum())},
        {"item": "Eligible censored", "value": int((eligible["event_observed"] == 0).sum())},
        {"item": "Time-invariant covariates", "value": len(BASELINE_COVARIATES)},
        {"item": "Time-varying covariates", "value": len(TV_NAMES)},
        {"item": "Total value channels per wave", "value": len(BASELINE_COVARIATES) + len(TV_NAMES)},
        {"item": "Assessment years since baseline", "value": ",".join(f"{x:.2f}" for x in ASSESSMENT_YEARS)},
        {"item": "Minimum history visits", "value": args.min_history_visits},
        {"item": "Prediction horizons", "value": ",".join(map(str, args.horizons))},
        {"item": "Event-time MAE unit", "value": "years"},
        {"item": "Transformer beta grid", "value": ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in args.w_grid)},
        {"item": "Transformer d_model", "value": TRANSFORMER_D_MODEL},
        {"item": "Transformer heads", "value": TRANSFORMER_NHEAD},
        {"item": "Transformer FFN", "value": TRANSFORMER_FFN},
        {"item": "Transformer dropout", "value": TRANSFORMER_DROPOUT},
        {"item": "Transformer weight decay", "value": TRANSFORMER_WEIGHT_DECAY},
        {"item": "Transformer layers", "value": TRANSFORMER_LAYERS},
        {"item": "Transformer maximum epochs", "value": args.epochs},
        {"item": "Cox specification", "value": "simulation-aligned start-stop time-dependent Cox; landmark-aligned evaluation"},
        {"item": "Cox ridge", "value": COX_RIDGE},
        {"item": "Cox ridge tuned", "value": "No"},
        {"item": "Transformer beta selection metrics", "value": "Risk C-index@3;Risk C-index@5;Brier@3;Brier@5;Binary NLL@3;Binary NLL@5;ECE@3;ECE@5"},
        {"item": "Primary CV summary", "value": "mean of fold-specific outer-test metrics; SD and SE also reported"},
    ])
    audit.to_csv(outdir / "01_data_audit.csv", index=False)

    inventory_rows = []
    for c in BASELINE_COVARIATES:
        inventory_rows.append({"type": "time-invariant", "stream": c, "waves_available": "T1"})
    for name in TV_NAMES:
        if name in TV_COLUMN_MAP:
            waves = sorted(TV_COLUMN_MAP[name])
            construction = "original longitudinal column"
        else:
            waves = sorted(COUPLE_TV_COLUMN_MAP[name])
            construction = "wave-specific husband/wife mean"
        inventory_rows.append({
            "type": "time-varying",
            "stream": name,
            "waves_available": ",".join(f"T{w}" for w in waves),
            "construction": construction,
        })
    pd.DataFrame(inventory_rows).to_csv(outdir / "00_covariate_inventory.csv", index=False)

    if args.audit_only:
        pp = fit_preprocessor(eligible)
        proc = transform_people(eligible, pp)
        lm = build_multiple_landmark_samples(proc, args.min_history_visits)
        print("Audit complete.")
        print(f"Original participants: {raw['n_original']}")
        print(f"Eligible participants: {len(eligible['ids'])}")
        print(f"Eligible events: {int(eligible['event_observed'].sum())}")
        print(f"Time-invariant covariates: {len(BASELINE_COVARIATES)}")
        print(f"Time-varying streams: {len(TV_NAMES)}")
        print(f"Total landmark records: {len(lm['event_time'])}")
        return

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = eligible["event_observed"]
    fold_assign = []
    all_fold_metrics = []
    all_horizon_preds = []
    all_event_preds = []
    hypers = []
    beta_details = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n===== OUTER FOLD {fold}/{args.folds} =====", flush=True)
        for i in tr:
            fold_assign.append({"person_id": str(eligible["ids"][i]), "fold": fold, "role": "train"})
        for i in te:
            fold_assign.append({"person_id": str(eligible["ids"][i]), "fold": fold, "role": "test"})
        fm, hpr, epr, hp, bd = run_fold(eligible, tr, te, fold, args.seed, args)
        all_fold_metrics.extend(fm)
        all_horizon_preds.append(hpr)
        all_event_preds.append(epr)
        hypers.append(hp)
        beta_details.extend(bd)

    pd.DataFrame(fold_assign).to_csv(outdir / "02_fold_assignments.csv", index=False)
    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_csv(outdir / "03_fold_metrics.csv", index=False)

    horizon_pred = pd.concat(all_horizon_preds, ignore_index=True)
    event_pred = pd.concat(all_event_preds, ignore_index=True)
    horizon_pred.to_csv(outdir / "04_oof_horizon_predictions.csv", index=False)
    event_pred.to_csv(outdir / "05_oof_event_time_predictions.csv", index=False)

    fold_summary = summarize_fold_metrics(fold_metrics_df)
    fold_summary.to_csv(outdir / "06_fold_performance_summary.csv", index=False)
    make_wide_from_fold_summary(fold_summary).to_csv(outdir / "07_method_comparison_wide.csv", index=False)

    pooled = pooled_metrics_from_predictions(horizon_pred, event_pred)
    pooled.to_csv(outdir / "08_pooled_oof_performance.csv", index=False)
    make_wide_from_pooled(pooled).to_csv(outdir / "09_pooled_oof_comparison_wide.csv", index=False)

    pd.DataFrame(hypers).to_csv(outdir / "10_selected_hyperparameters.csv", index=False)
    pd.DataFrame(beta_details).to_csv(outdir / "11_transformer_beta_selection_details.csv", index=False)

    with open(outdir / "analysis_settings.json", "w") as f:
        json.dump({
            "analysis_condition": "S-FS 5-TV predictors + actual irregular assessment times + R1 Transformer + simulation-aligned Cox",
            "data": os.path.abspath(args.data),
            "seed": args.seed,
            "folds": args.folds,
            "inner_val_fraction": args.inner_val_fraction,
            "min_history_visits": args.min_history_visits,
            "horizons": args.horizons,
            "assessment_years_since_baseline": ASSESSMENT_YEARS.tolist(),
            "horizon_definition": "wave-based (+3 and +5 waves); actual mean years used for time encoding and continuous-time Cox/RDSM durations",
            "event_time_mae_unit": "years",
            "transformer": {
                "d_model": TRANSFORMER_D_MODEL,
                "nhead": TRANSFORMER_NHEAD,
                "ffn": TRANSFORMER_FFN,
                "dropout": TRANSFORMER_DROPOUT,
                "weight_decay": TRANSFORMER_WEIGHT_DECAY,
                "layers": TRANSFORMER_LAYERS,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "w_grid": args.w_grid,
                "beta_selection_rule": "minimum average validation rank",
                "beta_selection_metrics": [
                    "Risk C-index@3", "Risk C-index@5",
                    "Brier@3", "Brier@5",
                    "Binary NLL@3", "Binary NLL@5",
                    "ECE@3", "ECE@5"
                ],
                "beta_selection_excludes": ["AUC", "MAE_event_time", "EventTime_C-index"]
            },
            "primary_cv_summary": "mean of fold-specific outer-test metrics with SD and SE",
            "secondary_cv_summary": "pooled out-of-fold metrics",
            "cox": {"ridge": COX_RIDGE, "ridge_tuned": False, "type": "simulation-aligned start-stop time-dependent Cox; landmark-aligned prediction using current covariates"},
            "rdsm": {"k": args.rdsm_k, "hidden": args.rdsm_hidden, "layers": args.rdsm_layers, "type": args.rdsm_type, "distribution": args.rdsm_distribution, "iters": args.rdsm_iters, "lr": args.rdsm_lr, "batch_size": args.rdsm_batch_size},
            "baseline_covariates": BASELINE_COVARIATES,
            "time_varying_covariates": TV_NAMES,
        }, f, indent=2)

    print("\nDone. Main comparison table:", flush=True)
    print((outdir / "07_method_comparison_wide.csv").resolve(), flush=True)

if __name__ == "__main__":
    main()
