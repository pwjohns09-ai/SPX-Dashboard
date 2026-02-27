"""
Analog Engine Backtest (v3)
===========================
For each trading day in the past year, this script:
  1. Pretends that day is "today"
  2. Runs the analog engine (finds top 20 matches using only PRIOR data)
  3. Records the predicted prob_up and median return for each horizon
  4. Compares against what ACTUALLY happened
  5. Outputs accuracy report + saves results to backtest_results.json

Z-score normalization is active: compute_regime_zstats() is called per
walk-forward window slice for consistent calibration with inference.

Usage:
    python backtest_analog.py
"""

import json
import math
import sys
import statistics
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Import the engine functions from v3/calculate_indicator.py
sys.path.insert(0, str(Path(__file__).parent))
from calculate_indicator import (
    load_data, compute_all_features, SCALAR_FEATURES, MIN_HISTORY,
    normalize_pattern, cosine_distance, sma, rolling_std, MAX_MATCHES,
    MAX_MATCHES_BY_HORIZON, HORIZON_WEIGHTS, ENSEMBLE_LOOKBACKS,
    ENSEMBLE_LOOKBACKS_BY_HORIZON, compute_regime_zstats
)

BASE_DIR = Path(__file__).parent
LOOKBACK = 10
MATCH_COUNT = 20
BACKTEST_DAYS = 9999  # use all available history (clamped by MIN_POOL_SIZE)
MIN_POOL_SIZE = 1500  # minimum prior trading days needed before starting backtest
FORWARD_HORIZONS = [1, 5, 10, 21, 42, 63]


def precompute_data(df, lookback):
    """
    Called ONCE before the backtest loop.
    Pre-builds the pattern matrix and feature matrix as numpy arrays so
    prepare_analogs_at can use fully vectorized operations instead of
    per-candidate Python loops.
    """
    n = len(df)
    daily_ret = df["daily_ret"].values

    # Pre-compute normalized return patterns for every day
    pattern_matrix = np.zeros((n, lookback), dtype=np.float64)
    for i in range(lookback - 1, n):
        pat = daily_ret[i - lookback + 1: i + 1]
        m, s = pat.mean(), pat.std()
        pattern_matrix[i] = (pat - m) / s if s > 0 else 0.0

    # Pre-extract scalar features (all except pattern_distance) as a 2-D array
    other_cols = [c for c in SCALAR_FEATURES if c != "pattern_distance"]
    feature_matrix = np.column_stack([df[c].values for c in other_cols]).astype(np.float64)
    above_200ma_arr = df["above_200ma"].values

    return pattern_matrix, feature_matrix, above_200ma_arr


def prepare_analogs_at(target_idx, lookback, pattern_matrix, feature_matrix, above_200ma_arr,
                       zscore_stats=None):
    """
    Vectorized version — run ONCE per target day.
    Returns (target_vec, cand_indices, cand_matrix) fully scaled, or None.
    All heavy lifting is numpy matrix ops — no Python loops over candidates.
    """
    min_idx = max(lookback, MIN_HISTORY)
    max_idx = target_idx - 1   # widest possible window; per-horizon filtering in score_analogs
    if max_idx < min_idx:
        return None

    cand_range = np.arange(min_idx, max_idx + 1)

    # Regime filter (vectorized)
    target_regime = above_200ma_arr[target_idx]
    if not np.isnan(target_regime):
        cand_regimes = above_200ma_arr[cand_range]
        mask = (cand_regimes == target_regime) & ~np.isnan(cand_regimes)
        cand_range = cand_range[mask]
    if len(cand_range) == 0:
        return None

    # Vectorized cosine distance: all candidates vs target in one matrix op
    target_pat = pattern_matrix[target_idx]          # (lookback,)
    cand_pats  = pattern_matrix[cand_range]          # (n_cands, lookback)
    dots       = cand_pats @ target_pat              # (n_cands,)
    norm_t     = np.linalg.norm(target_pat)
    norms_c    = np.linalg.norm(cand_pats, axis=1)  # (n_cands,)
    denom      = norms_c * norm_t
    denom[denom == 0] = 1e-10
    pattern_dists = 1.0 - (dots / denom)            # (n_cands,)

    # Build full feature matrix: [pattern_dist | other_features]
    cand_other  = feature_matrix[cand_range]         # (n_cands, n_other)
    target_other = feature_matrix[target_idx]        # (n_other,)
    cand_full   = np.column_stack([pattern_dists, cand_other])   # (n_cands, n_feat)
    target_full = np.concatenate([[0.0], target_other])          # (n_feat,)

    # Drop candidates with any NaN
    valid = ~np.any(np.isnan(cand_full), axis=1)
    cand_full   = cand_full[valid]
    cand_indices = cand_range[valid].tolist()
    if len(cand_full) == 0:
        return None

    if zscore_stats is not None:
        # Regime-aware z-score normalization (matches calculate_indicator.py)
        target_regime = above_200ma_arr[target_idx]
        regime_key    = float(int(target_regime)) if not np.isnan(target_regime) else 1.0
        zstds         = zscore_stats.get(regime_key, zscore_stats.get(1.0))
        diff          = cand_full - target_full                             # (n_cands, n_feat)
        cand_matrix   = np.clip(diff / zstds, -4.0, 4.0).astype(np.float32)
        target_vec    = np.zeros(len(zstds), dtype=np.float32)
    else:
        # Original min-max scaling
        all_rows  = np.vstack([cand_full, target_full])
        feat_min  = all_rows.min(axis=0)
        feat_max  = all_rows.max(axis=0)
        feat_rng  = feat_max - feat_min
        feat_rng[feat_rng == 0] = 1.0
        scaled    = (all_rows - feat_min) / feat_rng
        cand_matrix = scaled[:-1].astype(np.float32)
        target_vec  = scaled[-1].astype(np.float32)

    return target_vec, cand_indices, cand_matrix


def score_analogs(target_vec, cand_indices, cand_matrix, feature_weights, match_count,
                  max_candidate_idx=None, max_matches=MAX_MATCHES):
    """
    Fast step — run once per horizon per target day.
    Applies feature weights and returns top analog matches using
    vectorized numpy distance computation.
    """
    # Per-horizon candidate window: exclude candidates too recent for this horizon
    if max_candidate_idx is not None:
        mask = np.array(cand_indices) <= max_candidate_idx
        cand_indices = [idx for idx, m in zip(cand_indices, mask) if m]
        cand_matrix = cand_matrix[mask]
        if len(cand_indices) == 0:
            return []

    if feature_weights is not None:
        w = np.array([feature_weights.get(k, 1.0) for k in SCALAR_FEATURES], dtype=np.float32)
    else:
        w = np.ones(len(SCALAR_FEATURES), dtype=np.float32)

    # Vectorized weighted Euclidean distance
    t_vec = target_vec * w
    c_mat = cand_matrix * w                      # (n_cands, n_features)
    dists = np.sqrt(((c_mat - t_vec) ** 2).sum(axis=1))  # (n_cands,)

    order = np.argsort(dists)
    sorted_dists = dists[order]
    sorted_indices = [cand_indices[i] for i in order]

    # Take the closest max_matches candidates (pure top-N)
    qualified = list(zip(sorted_indices, sorted_dists.tolist()))[:max_matches]

    if len(qualified) < match_count:
        return []

    raw_weights = [1.0 / (d + 0.0001) for _, d in qualified]
    total_w = sum(raw_weights)
    return [(idx, d, w / total_w) for (idx, d), w in zip(qualified, raw_weights)]


def backtest():
    print("Loading data...")
    df = load_data()
    n = len(df)
    closes = df["Close"].values
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()

    print("Computing features for full dataset...")
    compute_all_features(df, LOOKBACK)

    print("Precomputing pattern and feature matrices...")
    pattern_matrices = {}
    feature_matrix = above_200ma_arr = None
    all_lookbacks = set()
    for lbs in ENSEMBLE_LOOKBACKS_BY_HORIZON.values():
        all_lookbacks.update(lbs)
    for lb in sorted(all_lookbacks):
        pm, fm, a200 = precompute_data(df, lb)
        pattern_matrices[lb] = pm
        if feature_matrix is None:
            feature_matrix, above_200ma_arr = fm, a200

    print("Computing regime z-score stats...")
    zscore_stats = compute_regime_zstats(df)

    # Backtest period: last ~504 trading days (2 years)
    # But we need 63 forward days to verify, so end 63 days before last day
    bt_end = n - 1 - 63  # last day we can verify 63-day forward
    bt_start = bt_end - BACKTEST_DAYS + 1
    if bt_start < MIN_POOL_SIZE:
        bt_start = MIN_POOL_SIZE  # ensure engine has enough history to find quality matches

    bt_days = list(range(bt_start, bt_end + 1))
    print(f"Backtest period: {dates[bt_days[0]]} to {dates[bt_days[-1]]} ({len(bt_days)} days)")
    print(f"Forward verification through: {dates[min(bt_days[-1] + 63, n - 1)]}")
    print()

    # Track results per horizon
    horizon_results = {h: [] for h in FORWARD_HORIZONS}

    for i, target_idx in enumerate(bt_days):
        if i % 25 == 0:
            pct = i / len(bt_days) * 100
            print(f"  Processing {dates[target_idx]} ({pct:.0f}%)...")

        # Prepare once per lookback per day (vectorized)
        preps = {}
        for lb in all_lookbacks:
            p = prepare_analogs_at(target_idx, lb, pattern_matrices[lb],
                                   feature_matrix, above_200ma_arr,
                                   zscore_stats=zscore_stats)
            if p is not None:
                preps[lb] = p
        if not preps:
            continue

        for h in FORWARD_HORIZONS:
            fwd_idx = target_idx + h
            if fwd_idx >= n:
                continue

            # Score each lookback for this horizon and combine matches
            h_weights = HORIZON_WEIGHTS.get(h)
            mm = MAX_MATCHES_BY_HORIZON.get(h, MAX_MATCHES)
            h_lookbacks = ENSEMBLE_LOOKBACKS_BY_HORIZON.get(h, list(all_lookbacks))
            all_matches = []
            for lb in h_lookbacks:
                if lb not in preps:
                    continue
                target_vec, cand_indices, cand_matrix = preps[lb]
                m = score_analogs(target_vec, cand_indices, cand_matrix,
                                  h_weights, MATCH_COUNT,
                                  max_candidate_idx=target_idx - h - 1,
                                  max_matches=mm)
                if m:
                    all_matches.extend(m)
            if not all_matches:
                continue

            # Re-normalize combined weights
            total_w = sum(w for _, _, w in all_matches)
            matches = [(idx, d, w / total_w) for idx, d, w in all_matches]

            # Actual return
            actual_ret = (closes[fwd_idx] / closes[target_idx] - 1) * 100
            actual_up = actual_ret > 0

            # Predicted: similarity-weighted forward returns of analogs
            analog_fwd_pairs = []
            for idx, dist, sim in matches:
                af_idx = idx + h
                if af_idx < target_idx:  # only use data that existed at that time
                    ret = (closes[af_idx] / closes[idx] - 1) * 100
                    analog_fwd_pairs.append((ret, sim))

            if not analog_fwd_pairs:
                continue

            total_weight = sum(w for _, w in analog_fwd_pairs)
            if not total_weight or not math.isfinite(total_weight):
                continue
            pred_prob_up = sum(w for r, w in analog_fwd_pairs if r > 0) / total_weight * 100
            if not math.isfinite(pred_prob_up):
                continue
            analog_fwd_rets = [r for r, w in analog_fwd_pairs]
            pred_median = statistics.median(analog_fwd_rets)
            pred_mean = sum(r * w for r, w in analog_fwd_pairs) / total_weight

            horizon_results[h].append({
                "date": dates[target_idx],
                "actual_ret": round(actual_ret, 2),
                "actual_up": actual_up,
                "pred_prob_up": round(pred_prob_up, 1),
                "pred_median": round(pred_median, 2),
                "pred_mean": round(pred_mean, 2),
                "n_analogs": len(analog_fwd_rets),
            })

    # -- Generate Report --
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - Analog Engine v3 (regime z-score)")
    print(f"Period: {dates[bt_days[0]]} to {dates[bt_days[-1]]} ({len(bt_days)} trading days)")
    print(f"Settings: lookback={LOOKBACK}, matches={MATCH_COUNT}")
    print("=" * 70)

    def norm_sf(z):
        """One-tailed survival function of standard normal (P(Z > z))."""
        return 0.5 * math.erfc(z / math.sqrt(2))

    def pearson_corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx * dy > 0 else 0.0

    report = {}
    for h in FORWARD_HORIZONS:
        results = horizon_results[h]
        if not results:
            continue

        results_sorted = sorted(results, key=lambda r: r["date"])
        total = len(results_sorted)
        actual_up_count = sum(1 for r in results_sorted if r["actual_up"])
        actual_up_rate  = actual_up_count / total * 100

        bullish_calls = [r for r in results_sorted if r["pred_prob_up"] > 50]
        bearish_calls = [r for r in results_sorted if r["pred_prob_up"] < 50]

        bull_correct = sum(1 for r in bullish_calls if r["actual_up"]) if bullish_calls else 0
        bear_correct = sum(1 for r in bearish_calls if not r["actual_up"]) if bearish_calls else 0

        bull_accuracy = (bull_correct / len(bullish_calls) * 100) if bullish_calls else 0
        bear_accuracy = (bear_correct / len(bearish_calls) * 100) if bearish_calls else 0

        overall_correct = bull_correct + bear_correct
        overall_calls   = len(bullish_calls) + len(bearish_calls)
        overall_accuracy = (overall_correct / overall_calls * 100) if overall_calls else 0

        direction_match = sum(1 for r in results_sorted
                              if (r["pred_median"] > 0) == r["actual_up"]) / total * 100
        mae = statistics.mean(abs(r["pred_median"] - r["actual_ret"]) for r in results_sorted)

        # ── Brier score ───────────────────────────────────────────────────────
        brier       = sum((r["pred_prob_up"] / 100 - int(r["actual_up"])) ** 2
                          for r in results_sorted) / total
        brier_naive = (actual_up_rate / 100) * (1 - actual_up_rate / 100)
        brier_skill = 1 - brier / brier_naive if brier_naive > 0 else 0.0

        # ── Information Coefficient (Pearson r) ───────────────────────────────
        preds   = [r["pred_prob_up"] for r in results_sorted]
        actuals = [int(r["actual_up"])  for r in results_sorted]
        ic = pearson_corr(preds, actuals)

        # ── Binomial significance test (H0: p=0.5) ───────────────────────────
        z_stat  = ((overall_correct - overall_calls * 0.5) /
                   math.sqrt(overall_calls * 0.5 * 0.5))
        p_value = norm_sf(z_stat)
        margin  = 1.96 * math.sqrt(overall_accuracy / 100 *
                                    (1 - overall_accuracy / 100) / overall_calls)
        ci_lo   = max(0.0, overall_accuracy / 100 - margin) * 100
        ci_hi   = min(1.0, overall_accuracy / 100 + margin) * 100

        # ── Rolling 252-day accuracy (sampled every 5 days) ───────────────────
        window = 252
        rolling_acc = []
        for i in range(window - 1, len(results_sorted), 5):
            w = results_sorted[i - window + 1: i + 1]
            n_bull_w = [r for r in w if r["pred_prob_up"] > 50]
            n_bear_w = [r for r in w if r["pred_prob_up"] < 50]
            n_corr_w = (sum(1 for r in n_bull_w if r["actual_up"]) +
                        sum(1 for r in n_bear_w if not r["actual_up"]))
            n_call_w = len(n_bull_w) + len(n_bear_w)
            rolling_acc.append({
                "date":     results_sorted[i]["date"],
                "accuracy": round(n_corr_w / n_call_w * 100, 1) if n_call_w else 0,
            })

        # ── Recent holdout: 2022-01-01 onward ─────────────────────────────────
        recent = [r for r in results_sorted if r["date"] >= "2022-01-01"]
        recent_stats = None
        if recent:
            r_bull = [r for r in recent if r["pred_prob_up"] > 50]
            r_bear = [r for r in recent if r["pred_prob_up"] < 50]
            r_corr = (sum(1 for r in r_bull if r["actual_up"]) +
                      sum(1 for r in r_bear if not r["actual_up"]))
            r_call = len(r_bull) + len(r_bear)
            recent_stats = {
                "start":            "2022-01-01",
                "samples":          len(recent),
                "overall_accuracy": round(r_corr / r_call * 100, 1) if r_call else 0,
                "actual_up_rate":   round(sum(1 for r in recent if r["actual_up"])
                                          / len(recent) * 100, 1),
                "brier_score":      round(sum((r["pred_prob_up"] / 100 - int(r["actual_up"])) ** 2
                                              for r in recent) / len(recent), 4),
            }

        # ── Coarse calibration (original 4 buckets) ───────────────────────────
        buckets = {"0-30%": [], "30-50%": [], "50-70%": [], "70-100%": []}
        for r in results_sorted:
            p = r["pred_prob_up"]
            if   p < 30: buckets["0-30%"].append(r)
            elif p < 50: buckets["30-50%"].append(r)
            elif p < 70: buckets["50-70%"].append(r)
            else:        buckets["70-100%"].append(r)

        # ── Fine calibration (5pp buckets) ────────────────────────────────────
        fine_edges = list(range(30, 75, 5)) + [75]
        fine_buckets = {}
        for lo, hi in zip(fine_edges[:-1], fine_edges[1:]):
            key = f"{lo}-{hi}%"
            fine_buckets[key] = [r for r in results_sorted if lo <= r["pred_prob_up"] < hi]
        fine_buckets["75%+"] = [r for r in results_sorted if r["pred_prob_up"] >= 75]

        # ── Print ─────────────────────────────────────────────────────────────
        print(f"\n{'-' * 70}")
        print(f"  {h}-DAY FORWARD HORIZON")
        print(f"{'-' * 70}")
        print(f"  Samples: {total}")
        print(f"  Actual up rate: {actual_up_rate:.1f}%")
        print(f"  Bullish calls (>50%): {len(bullish_calls)} | accuracy: {bull_accuracy:.1f}%")
        print(f"  Bearish calls (<50%): {len(bearish_calls)} | accuracy: {bear_accuracy:.1f}%")
        print(f"  Overall directional accuracy: {overall_accuracy:.1f}%  "
              f"[95% CI: {ci_lo:.1f}%-{ci_hi:.1f}%  p={p_value:.2e}]")
        print(f"  Median prediction direction match: {direction_match:.1f}%")
        print(f"  Mean absolute error (median pred vs actual): {mae:.2f}%")
        print(f"  Brier score: {brier:.4f}  (naive: {brier_naive:.4f}  "
              f"skill: {brier_skill:+.4f})")
        print(f"  IC (Pearson): {ic:.4f}")
        if recent_stats:
            print(f"  Recent holdout (2022+): {recent_stats['overall_accuracy']:.1f}% "
                  f"({recent_stats['samples']} days)")
        print()
        print(f"  Calibration (predicted prob -> actual up rate):")
        for bname, bdata in buckets.items():
            if bdata:
                bup = sum(1 for r in bdata if r["actual_up"]) / len(bdata) * 100
                ap  = statistics.mean(r["pred_prob_up"] for r in bdata)
                print(f"    {bname:>8s}: {len(bdata):4d} days | "
                      f"avg predicted: {ap:.0f}% | actual up: {bup:.1f}%")
            else:
                print(f"    {bname:>8s}: no samples")
        print()
        print(f"  Fine calibration (5pp buckets):")
        for bname, bdata in fine_buckets.items():
            if bdata:
                bup = sum(1 for r in bdata if r["actual_up"]) / len(bdata) * 100
                ap  = statistics.mean(r["pred_prob_up"] for r in bdata)
                bar = "#" * int(bup / 5)
                print(f"    {bname:>8s}: {len(bdata):4d} days | "
                      f"pred {ap:.0f}% -> actual {bup:.1f}%  {bar}")
            else:
                print(f"    {bname:>8s}: no samples")

        report[str(h)] = {
            "horizon_days":      h,
            "total_samples":     total,
            "actual_up_rate":    round(actual_up_rate, 1),
            "bullish_calls":     len(bullish_calls),
            "bullish_accuracy":  round(bull_accuracy, 1),
            "bearish_calls":     len(bearish_calls),
            "bearish_accuracy":  round(bear_accuracy, 1),
            "overall_accuracy":  round(overall_accuracy, 1),
            "direction_match":   round(direction_match, 1),
            "mae":               round(mae, 2),
            "brier_score":       round(brier, 4),
            "brier_naive":       round(brier_naive, 4),
            "brier_skill":       round(brier_skill, 4),
            "ic":                round(ic, 4),
            "p_value":           round(p_value, 6),
            "z_stat":            round(z_stat, 3),
            "ci_95":             [round(ci_lo, 1), round(ci_hi, 1)],
            "recent_holdout":    recent_stats,
            "rolling_accuracy":  rolling_acc,
            "calibration": {
                name: {
                    "count":          len(data),
                    "avg_predicted":  round(statistics.mean(
                                          r["pred_prob_up"] for r in data), 1) if data else 0,
                    "actual_up_rate": round(sum(1 for r in data if r["actual_up"])
                                           / len(data) * 100, 1) if data else 0,
                } for name, data in buckets.items()
            },
            "fine_calibration": {
                name: {
                    "count":          len(data),
                    "avg_predicted":  round(statistics.mean(
                                          r["pred_prob_up"] for r in data), 1) if data else 0,
                    "actual_up_rate": round(sum(1 for r in data if r["actual_up"])
                                           / len(data) * 100, 1) if data else 0,
                } for name, data in fine_buckets.items()
            },
            "calibration_pairs": [
                {"pred": r["pred_prob_up"], "actual_up": int(r["actual_up"])}
                for r in results_sorted
            ],
        }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Horizon':>10s} | {'Samples':>7s} | {'Bull Acc':>8s} | {'Bear Acc':>8s} | "
          f"{'Overall':>8s} | {'Dir Match':>9s} | {'MAE':>6s} | {'Brier':>7s} | {'IC':>7s}")
    print(f"{'-' * 70}")
    for h in FORWARD_HORIZONS:
        r = report.get(str(h))
        if r:
            print(f"{h:>7d}d   | {r['total_samples']:>7d} | "
                  f"{r['bullish_accuracy']:>7.1f}% | {r['bearish_accuracy']:>7.1f}% | "
                  f"{r['overall_accuracy']:>7.1f}% | {r['direction_match']:>8.1f}% | "
                  f"{r['mae']:>5.2f}% | {r['brier_score']:>7.4f} | {r['ic']:>7.4f}")

    # Save results
    output_file = BASE_DIR / "backtest_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "backtest_period": {
                "start": dates[bt_days[0]],
                "end":   dates[bt_days[-1]],
                "days":  len(bt_days),
            },
            "config": {
                "lookback":      LOOKBACK,
                "match_count":   MATCH_COUNT,
                "normalization": "regime_zscore",
            },
            "horizons": report,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    backtest()
