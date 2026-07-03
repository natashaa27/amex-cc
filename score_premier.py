"""
Amex Campus Challenge 2026 — Round 1
Profitability scoring pipeline for the Premier Card.

Builds five philosophy-based scores (S1..S5), emits a ranked Prediction per
cardmember, and validates the ranking the way an UNSUPERVISED, parameter-free
score must be validated:
  (A) philosophy agreement   — do the 5 philosophies rank customers alike?
  (B) assumption sensitivity — does the top-20% set survive rate perturbation?

Usage:
    python score_premier.py --data book1_clean.csv --out scores_out

Economic assumptions (all documented, all overridable) live in ASSUMPTIONS.
Because the leaderboard metric is ordinal (top-20% overlap), the SUBMITTED score
is S5 = mean of the percentile-ranks of S1..S4 (robust consensus). S4 (dollar
unit-economics) is reported alongside as the explainable centrepiece.
"""

import argparse, os, json
import numpy as np, pandas as pd

FEATS = [f"f{i}" for i in range(1, 24)]

# ---- documented economic assumptions (net margins, issuer-cost view) --------
ASSUMPTIONS = dict(
    m_interchange = 0.018,     # net interchange take-rate
    cat_wt = dict(f7=1.0, f10=0.9, f8=0.8, f6=0.6, f9=0.6),  # net-of-rewards mix
    apr = 0.18,                # net interest margin on revolving balance
    cpp = 0.010,               # issuer cost per reward point redeemed ($)
    lounge_cost = 50.0,        # $ per lounge visit (f13)
    cab_cost = 15.0,           # $ per cab-credit month (f15)
    serv_cancel = 20.0,        # $ servicing per cancellation call (f2)
    serv_collect = 100.0,      # $ servicing per collection call (f3)
    lgd = 0.9,                 # loss given default
    lend_exposure_wt = 0.1,    # fraction of lend line treated as exposure
)


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    # informative missingness flags BEFORE imputation
    flags = pd.DataFrame({
        "flag_no_points":  df["f4"].isna().astype(int),
        "flag_no_spend":   df["f6"].isna().astype(int),
        "flag_no_lend":    df["f17"].isna().astype(int),
        "flag_no_click":   df["f23"].isna().astype(int),
    })
    X = df[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X["f7"] = X["f7"].clip(lower=0)            # floor negative "Other Spend"
    return df["id"], X, flags


def rank01(s):
    """percentile rank in [0,1]; robust to caps & skew."""
    return s.rank(pct=True)


def economics(X, A):
    """Return dollar-space revenue / cost / risk components (Series)."""
    interchange = A["m_interchange"] * sum(A["cat_wt"][c] * X[c] for c in A["cat_wt"])
    interest    = A["apr"] * X["f1"]
    rewards     = A["cpp"] * X["f21"]
    benefits    = X["f14"] + X["f16"] + A["lounge_cost"] * X["f13"] + A["cab_cost"] * X["f15"]
    servicing   = A["serv_cancel"] * X["f2"] + A["serv_collect"] * X["f3"]
    exposure    = X["f1"] + A["lend_exposure_wt"] * X["f17"]
    ecl         = X["f11"] * exposure * A["lgd"]
    return dict(interchange=interchange, interest=interest, rewards=rewards,
                benefits=benefits, servicing=servicing, ecl=ecl,
                revenue=interchange + interest,
                cost=rewards + benefits + servicing)


def build_scores(X, flags, A):
    e = economics(X, A)
    r = {c: rank01(X[c]) for c in FEATS}

    # engineered
    total_spend   = X[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1)
    risk_adj_int  = X["f1"] * (1 - r["f11"])
    distress      = 1 * X["f2"] + 3 * X["f3"]
    rel_depth     = X["f19"] + X["f20"]

    # ---- Philosophy scores (rank-space unless noted) ----
    S1 = (0.30 * rank01(e["revenue"]) + 0.20 * rank01(risk_adj_int)
          - 0.30 * r["f11"] - 0.25 * rank01(distress) - 0.15 * rank01(e["ecl"]))
    S2 = (0.40 * rank01(e["interchange"]) + 0.30 * rank01(e["interest"])
          + 0.20 * rank01(total_spend) - 0.10 * r["f11"])
    S3 = (0.25 * rank01(rel_depth) + 0.20 * r["f4"] + 0.20 * rank01(total_spend)
          + 0.10 * rank01(X["f12"] + X["f22"]) - 0.25 * rank01(distress))
    S4_dollar = e["revenue"] - e["cost"] - e["ecl"]          # faithful unit economics
    S4 = rank01(S4_dollar)

    # ---- Balanced consensus (submitted) ----
    S5 = pd.concat([rank01(S1), rank01(S2), rank01(S3), S4], axis=1).mean(axis=1)

    return pd.DataFrame({
        "S1_conservative": rank01(S1), "S2_revenue": rank01(S2),
        "S3_ltv": rank01(S3), "S4_unit_econ": S4, "S4_dollar": S4_dollar,
        "S5_balanced": S5,
    })


def top_pct_set(score, pct=0.20):
    k = int(len(score) * pct)
    return set(score.sort_values(ascending=False).index[:k])


def validate_agreement(scores):
    cols = ["S1_conservative", "S2_revenue", "S3_ltv", "S4_unit_econ", "S5_balanced"]
    print("\n=== (A) PHILOSOPHY AGREEMENT ===")
    print("Spearman rank-correlation between philosophies:")
    print(scores[cols].corr(method="spearman").round(2).to_string())
    print("\nJaccard overlap of top-20% sets:")
    tops = {c: top_pct_set(scores[c]) for c in cols}
    hdr = "        " + " ".join(f"{c[:8]:>8}" for c in cols)
    print(hdr)
    for a in cols:
        row = [f"{len(tops[a] & tops[b]) / len(tops[a] | tops[b]):.2f}" for b in cols]
        print(f"{a[:8]:>8} " + " ".join(f"{v:>8}" for v in row))


def validate_sensitivity(X, flags, base):
    print("\n=== (B) ASSUMPTION SENSITIVITY (S4 top-20% stability) ===")
    base_top = top_pct_set(base["S4_unit_econ"])
    perturb = {
        "apr -40%": ("apr", 0.6), "apr +40%": ("apr", 1.4),
        "cpp -50%": ("cpp", 0.5), "cpp +50%": ("cpp", 1.5),
        "interchange -30%": ("m_interchange", 0.7),
        "interchange +30%": ("m_interchange", 1.3),
        "lgd +30%": ("lgd", 1.3),
    }
    for name, (k, mult) in perturb.items():
        A = json.loads(json.dumps(ASSUMPTIONS))
        A[k] = ASSUMPTIONS[k] * mult
        s = build_scores(X, flags, A)["S4_unit_econ"]
        j = len(base_top & top_pct_set(s)) / len(base_top | top_pct_set(s))
        print(f"  {name:>18}:  top-20% Jaccard stability = {j:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="scores_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids, X, flags = load(args.data)
    print(f"Scoring {len(X):,} cardmembers")
    scores = build_scores(X, flags, ASSUMPTIONS)

    # submission: Prediction = balanced consensus score (continuous, rank-ordered)
    sub = pd.DataFrame({"ID": ids, "Prediction": scores["S5_balanced"].values})
    sub.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    scores.assign(id=ids.values).to_csv(os.path.join(args.out, "all_scores.csv"), index=False)

    validate_agreement(scores)
    validate_sensitivity(X, flags, scores)

    # who's in the top 20%? sanity profile
    top = top_pct_set(scores["S5_balanced"])
    mask = scores.index.isin(top)
    print("\n=== TOP-20% PROFILE (mean vs rest) ===")
    prof = pd.DataFrame({
        "top20_mean": X[mask].mean(), "rest_mean": X[~mask].mean()
    }).loc[["f1", "f7", "f10", "f11", "f3", "f21", "f13", "f19", "f17"]]
    prof["ratio"] = (prof.top20_mean / prof.rest_mean.replace(0, np.nan)).round(1)
    print(prof.round(1).to_string())
    print(f"\nWrote {args.out}/predictions.csv and all_scores.csv")


if __name__ == "__main__":
    main()
