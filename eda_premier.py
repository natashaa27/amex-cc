"""
Amex Campus Challenge 2026 — Round 1
Complete EDA pipeline for the Premier Card profitability dataset.

Usage:
    python eda_premier.py --data path/to/train.csv --out eda_out

Produces (in --out):
    01_summary_stats.csv        per-variable stats (missing, skew, zeros, outliers, ...)
    02_distributions.png        hist + log-hist for every feature
    03_boxplots.png             outlier view (robust z / IQR)
    04_correlation.png          Spearman + Pearson heatmaps
    05_missingness.png          missing-value matrix
    06_redundancy_checks.csv    identity/subset checks (f5 vs f6-f10, f17 vs f18, ...)
    07_importance_ranking.csv   EDA-only variable importance

No target column exists in R1, so "importance" is unsupervised:
it blends (a) alignment with a business-constructed profitability proxy,
(b) non-redundancy, and (c) raw signal (non-degeneracy).
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- Feature dictionary (from 6a3eb1af6b994_feature_description.csv) ----
DESC = {
    "f1":  "Avg Revolve Balance 12m",   "f2":  "Cancellation Calls 12m",
    "f3":  "Cancel Calls (Collection)", "f4":  "Rewards Points Balance",
    "f5":  "Total Spend 12m",           "f6":  "Airlines Spend 12m",
    "f7":  "Other Spend 12m",           "f8":  "Entertainment Spend 12m",
    "f9":  "Lodging Spend 12m",         "f10": "Dining Spend 12m",
    "f11": "Avg Risk Score 12m",        "f12": "Website Login Counts",
    "f13": "Lounge Access Count",       "f14": "Airline Credits Used",
    "f15": "Cab Benefits Usage",        "f16": "Entertainment Credit Used",
    "f17": "Total Lend Line",           "f18": "Total Consumer Lend Line",
    "f19": "# Supplementary Accounts",  "f20": "# Active Charge Cards",
    "f21": "Points Redeemed 12m",       "f22": "Emails Opened 6m",
    "f23": "Emails Clicked 6m",
}
SPEND_PARTS = ["f6", "f7", "f8", "f9", "f10"]          # should sum ~= f5
COST_BENEFITS = ["f13", "f14", "f15", "f16", "f21"]    # direct give-back / rewards cost
RISK_NEG = ["f2", "f3", "f11"]                          # churn / distress / credit risk


def load(path):
    df = pd.read_csv(path)
    # normalise the id column name
    for c in ["id", "ID", "unique_identifier", "Unnamed: 0"]:
        if c in df.columns:
            df = df.rename(columns={c: "id"})
            break
    feats = [c for c in df.columns if c != "id"]
    return df, feats


def summary_stats(df, feats):
    rows = []
    n = len(df)
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_iqr = ((s < lo) | (s > hi)).sum()
        rows.append({
            "feature": f, "meaning": DESC.get(f, f),
            "n_missing": s.isna().sum(),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "pct_zero": round(100 * (s == 0).mean(), 2),
            "n_negative": (s < 0).sum(),
            "nunique": s.nunique(),
            "mean": s.mean(), "median": s.median(), "std": s.std(),
            "min": s.min(), "p99": s.quantile(.99), "max": s.max(),
            "skew": s.skew(), "kurtosis": s.kurt(),
            "pct_outlier_iqr": round(100 * out_iqr / n, 2),
        })
    return pd.DataFrame(rows)


def plot_distributions(df, feats, out):
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, f in zip(axes.ravel(), feats):
        s = pd.to_numeric(df[f], errors="coerce").dropna()
        # log1p if heavily right-skewed and non-negative
        use_log = s.min() >= 0 and s.skew() > 2
        vals = np.log1p(s) if use_log else s
        ax.hist(vals, bins=60, color="#006FCF")
        ax.set_title(f"{f}: {DESC.get(f,'')}" + (" (log1p)" if use_log else ""), fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_distributions.png"), dpi=110)
    plt.close(fig)


def plot_boxplots(df, feats, out):
    fig, ax = plt.subplots(figsize=(14, 6))
    # robust-scale each feature so they share an axis
    data = []
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        med, iqr = s.median(), (s.quantile(.75) - s.quantile(.25)) or 1
        data.append(((s - med) / iqr).dropna())
    try:
        ax.boxplot(data, tick_labels=feats, showfliers=True,
                   flierprops=dict(marker=".", markersize=2))
    except TypeError:  # older matplotlib
        ax.boxplot(data, labels=feats, showfliers=True,
                   flierprops=dict(marker=".", markersize=2))
    ax.set_title("Robust-scaled boxplots (outlier view)")
    ax.set_ylabel("(x - median) / IQR")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03_boxplots.png"), dpi=110)
    plt.close(fig)


def plot_corr(df, feats, out):
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, method in zip(axes, ["pearson", "spearman"]):
        c = num.corr(method=method)
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
        ax.set_title(f"{method.title()} correlation")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04_correlation.png"), dpi=110)
    plt.close(fig)
    return num.corr(method="spearman")


def plot_missing(df, feats, out):
    miss = df[feats].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(miss)), miss.values * 100, color="#B3242B")
    ax.set_xticks(range(len(miss))); ax.set_xticklabels(miss.index, rotation=90, fontsize=7)
    ax.set_ylabel("% missing"); ax.set_title("Missingness by feature")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05_missingness.png"), dpi=110)
    plt.close(fig)


def redundancy_checks(df, feats, out):
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    checks = []
    # 1) does total spend equal the sum of category spends?
    if set(SPEND_PARTS + ["f5"]).issubset(num.columns):
        recon = num[SPEND_PARTS].sum(axis=1)
        diff = (num["f5"] - recon)
        checks.append({"check": "f5 == sum(f6..f10)",
                       "median_abs_diff": diff.abs().median(),
                       "pct_within_1pct": round(100 * (diff.abs() <= 0.01 * num["f5"].abs()).mean(), 2)})
    # 2) is consumer lend line a subset of total lend line?
    if {"f17", "f18"}.issubset(num.columns):
        checks.append({"check": "f18 <= f17 (consumer subset)",
                       "pct_true": round(100 * (num["f18"] <= num["f17"]).mean(), 2),
                       "spearman": num["f17"].corr(num["f18"], method="spearman")})
    # 3) points redeemed should not exceed a plausible multiple of balance
    if {"f4", "f21"}.issubset(num.columns):
        checks.append({"check": "corr(points_balance f4, redeemed f21)",
                       "spearman": num["f4"].corr(num["f21"], method="spearman")})
    # 4) collection calls should be <= total cancellation calls
    if {"f2", "f3"}.issubset(num.columns):
        checks.append({"check": "f3 <= f2 (collection subset of cancel)",
                       "pct_true": round(100 * (num["f3"] <= num["f2"]).mean(), 2)})
    out_df = pd.DataFrame(checks)
    out_df.to_csv(os.path.join(out, "06_redundancy_checks.csv"), index=False)
    return out_df


def business_profit_proxy(num):
    """
    A transparent, unit-economics proxy used ONLY to rank features by
    alignment. NOT the final submission score. Signs follow the framework:
      revenue  = interchange(spend) + interest(revolve, risk-adjusted)
      cost     = rewards + benefit give-backs
      penalty  = risk / distress
    All terms are rank-standardised (z on ranks) so scale doesn't dominate.
    """
    def rz(col):  # rank-z
        if col not in num.columns:
            return 0.0
        r = num[col].rank(pct=True)
        return (r - r.mean()) / (r.std() + 1e-9)

    revenue = 0.5 * rz("f7") + 0.3 * rz("f5") + 0.2 * (rz("f6") + rz("f9") + rz("f10"))
    interest = rz("f1") * (1 - 0.5 * (rz("f11") - rz("f11").min()))   # risk-attenuated revolve
    cost = 0.3 * (rz("f13") + rz("f14") + rz("f15") + rz("f16")) + 0.4 * rz("f21")
    penalty = 0.6 * rz("f11") + 0.3 * rz("f3") + 0.2 * rz("f2")
    stickiness = 0.2 * (rz("f19") + rz("f20"))
    return revenue + 0.8 * interest + stickiness - 0.7 * cost - 0.9 * penalty


def importance_ranking(df, feats, spearman, out):
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    proxy = business_profit_proxy(num)

    # (a) alignment with business proxy
    align = num.apply(lambda s: abs(s.corr(proxy, method="spearman"))).fillna(0)
    # (b) non-redundancy: 1 - mean |corr| with the other features
    absc = spearman.abs().copy()
    arr = absc.values.copy()
    np.fill_diagonal(arr, np.nan)
    absc = pd.DataFrame(arr, index=absc.index, columns=absc.columns)
    nonredund = 1 - absc.mean(axis=1)
    # (c) signal: not near-constant (share of non-modal, non-zero mass)
    signal = num.apply(lambda s: 1 - max(s.value_counts(normalize=True).max(), s.isna().mean()))

    rank = pd.DataFrame({
        "feature": feats,
        "meaning": [DESC.get(f, f) for f in feats],
        "align_proxy": align.values,
        "non_redundancy": nonredund.reindex(feats).values,
        "signal": signal.values,
    })
    # weighted composite
    for c in ["align_proxy", "non_redundancy", "signal"]:
        rank[c + "_n"] = (rank[c] - rank[c].min()) / (rank[c].max() - rank[c].min() + 1e-9)
    rank["importance"] = (0.55 * rank["align_proxy_n"]
                          + 0.25 * rank["non_redundancy_n"]
                          + 0.20 * rank["signal_n"])
    rank = rank.sort_values("importance", ascending=False).reset_index(drop=True)
    rank.to_csv(os.path.join(out, "07_importance_ranking.csv"), index=False)
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eda_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, feats = load(args.data)
    print(f"Loaded {len(df):,} rows x {len(feats)} features")

    stats = summary_stats(df, feats)
    stats.to_csv(os.path.join(args.out, "01_summary_stats.csv"), index=False)
    print("\n=== SUMMARY STATS ===")
    print(stats.to_string(index=False))

    plot_distributions(df, feats, args.out)
    plot_boxplots(df, feats, args.out)
    spearman = plot_corr(df, feats, args.out)
    plot_missing(df, feats, args.out)

    checks = redundancy_checks(df, feats, args.out)
    print("\n=== REDUNDANCY / SUSPICIOUS-VALUE CHECKS ===")
    print(checks.to_string(index=False))

    rank = importance_ranking(df, feats, spearman, args.out)
    print("\n=== EDA-ONLY IMPORTANCE RANKING ===")
    print(rank[["feature", "meaning", "importance",
                "align_proxy", "non_redundancy", "signal"]].to_string(index=False))
    print(f"\nAll artifacts written to: {args.out}/")


if __name__ == "__main__":
    main()
