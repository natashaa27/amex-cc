"""
Amex Campus Challenge 2026 — Round 1  |  LEAN leaderboard model.

Minimal additive rank-score. Everything is a percentile-rank in [0,1] so the
~2.6% winsorization caps and heavy skew don't matter. Only spend + (optional)
risk / interest / cost. No speculative rates, no decoy features.

The f5-vs-category spend question is UNRESOLVED in the data (rank corr 0.012,
top-20% overlap 0.15) -> it MUST be settled on the public leaderboard. Use
--spend to switch, and --model to add terms one at a time.

Usage (your 10-probe plan):
  python score_lean.py --data d.csv --spend f5    --model m0   # probe 1
  python score_lean.py --data d.csv --spend cat   --model m0   # probe 2
  python score_lean.py --data d.csv --spend blend --model m0   # probe 3
  python score_lean.py --data d.csv --spend <best> --model m1  # +risk
  python score_lean.py --data d.csv --spend <best> --model m2  # +interest
  python score_lean.py --data d.csv --spend <best> --model m3  # +cost
Tune a/b/c with --a --b --c in later probes.
"""
import argparse, os
import numpy as np, pandas as pd

FEATS = [f"f{i}" for i in range(1, 24)]


def rank01(s):
    return s.rank(pct=True)


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    X = df[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X["f7"] = X["f7"].clip(lower=0)
    return df["id"], X


def spend_series(X, which):
    f5 = rank01(X["f5"])
    cat = rank01(X[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1))
    return {"f5": f5, "cat": cat, "blend": 0.5 * f5 + 0.5 * cat}[which]


def score(X, spend_kind, model, a, b, c):
    SP = spend_series(X, spend_kind)
    s = SP.copy()
    risk = rank01(X["f11"])
    interest = rank01(X["f1"] * (1 - risk))
    cost = rank01(X["f21"] + X["f14"] + 50 * X["f13"] + 15 * X["f15"])
    if model in ("m1", "m2", "m3"):
        s = s - a * risk
    if model in ("m2", "m3"):
        s = s + b * interest
    if model == "m3":
        s = s - c * cost
    return rank01(s)   # final rank-normalize -> clean [0,1] Prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="lean_out")
    ap.add_argument("--spend", choices=["f5", "cat", "blend"], default="blend")
    ap.add_argument("--model", choices=["m0", "m1", "m2", "m3"], default="m0")
    ap.add_argument("--a", type=float, default=0.4, help="risk weight")
    ap.add_argument("--b", type=float, default=0.3, help="interest weight")
    ap.add_argument("--c", type=float, default=0.3, help="cost weight")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids, X = load(args.data)
    pred = score(X, args.spend, args.model, args.a, args.b, args.c)
    out = pd.DataFrame({"ID": ids, "Prediction": pred.values})
    tag = f"{args.spend}_{args.model}"
    path = os.path.join(args.out, f"pred_{tag}.csv")
    out.to_csv(path, index=False)
    print(f"[{tag}]  scored {len(out):,}  ->  {path}")
    print(f"  weights a(risk)={args.a} b(int)={args.b} c(cost)={args.c}")


if __name__ == "__main__":
    main()
