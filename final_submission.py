#!/usr/bin/env python3
"""
Amex Campus Challenge 2026 — Round 1
FINAL competition submission generator.

Final model = ENSEMBLE (rank-average) of two functional forms that agree on the
profitable core but fail differently, which minimises variance on the hidden
30% private leaderboard:

  Model A  (unit-economics, multiplicative risk gate):
      base   = 0.45*spend + 0.20*interest - 0.20*cost
      A      = base * (1 - 0.50*risk) * (1 + 0.15*rel) - 0.10*collections
  Model B  (lean additive spend model):
      B      = spend - 0.40*risk + 0.30*interest - 0.30*cost

  Prediction = rank01( 0.5*rank01(A) + 0.5*rank01(B) )   in (0,1], higher = more profitable

All inputs are percentile ranks (immune to the ~2.6% winsorization caps + skew).
Spend is a 0.5/0.5 blend of f5 and Σ(f6..f10) — these are orthogonal in the data
(rank corr ~0.01); with no labels we CANNOT know which the hidden target uses, so
the blend hedges. Resolve on the public leaderboard if submissions allow.

Run:
    python final_submission.py --data <official_data.csv> \
        --template 6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx \
        --out submission_final.xlsx
Executes end-to-end, validates, and writes submission_final.xlsx.
"""
import argparse, sys
import numpy as np
import pandas as pd

FEATS = [f"f{i}" for i in range(1, 24)]
CAT = ["f6", "f7", "f8", "f9", "f10"]

# ---- final coefficients (expert/AHP-set; leaderboard-tunable) ---------------
CFG = dict(alpha=0.45, beta=0.20, gamma=0.20, lam=0.50, rho=0.15, delta=0.10,
           a=0.40, b=0.30, c=0.30, lounge_cost=50.0, cab_cost=15.0)


# ============================================================ preprocessing ==
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "id" not in df.columns:
        for alt in ("ID", "unique_identifier", "Unnamed: 0"):
            if alt in df.columns:
                df = df.rename(columns={alt: "id"}); break
    assert "id" in df.columns, "input data must contain an 'id' column"
    return df


def clean(df: pd.DataFrame):
    """Impute missing -> 0 (genuine non-user), floor negative Other Spend,
    keep informative missingness flags."""
    X = df[FEATS].apply(pd.to_numeric, errors="coerce")
    flags = pd.DataFrame({
        "flag_no_points": df["f4"].isna().astype(int),
        "flag_no_spend":  df["f6"].isna().astype(int),
        "flag_no_lend":   df["f17"].isna().astype(int),
        "flag_no_click":  df["f23"].isna().astype(int),
    }, index=df.index)
    X = X.fillna(0.0)
    X["f7"] = X["f7"].clip(lower=0)
    return X, flags


# ============================================================ scoring ========
def rank01(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def engineer(X: pd.DataFrame) -> dict:
    spend = 0.5 * rank01(X["f5"]) + 0.5 * rank01(X[CAT].sum(axis=1))
    risk = rank01(X["f11"])
    interest = rank01(X["f1"] * (1 - risk))
    cost = rank01(X["f21"] + X["f14"] + CFG["lounge_cost"] * X["f13"]
                  + CFG["cab_cost"] * X["f15"])
    rel = rank01(X["f19"])
    return dict(spend=spend, risk=risk, interest=interest, cost=cost,
                rel=rel, collections=X["f3"].astype(float))


def final_score_raw(X: pd.DataFrame) -> pd.Series:
    """Raw ensemble score (pre-final-rank). Final rank is applied over the full
    template so unscored IDs can be placed in a clean bottom band."""
    e = engineer(X)
    base = CFG["alpha"] * e["spend"] + CFG["beta"] * e["interest"] - CFG["gamma"] * e["cost"]
    A = base * (1 - CFG["lam"] * e["risk"]) * (1 + CFG["rho"] * e["rel"]) - CFG["delta"] * e["collections"]
    B = e["spend"] - CFG["a"] * e["risk"] + CFG["b"] * e["interest"] - CFG["c"] * e["cost"]
    return 0.5 * rank01(A) + 0.5 * rank01(B)      # raw ensemble in (0,1]


# ============================================================ submission =====
FRAMEWORK = {
 "Variables Used":
  "Economic drivers only. Revenue: f5 & f6-f10 (spend->interchange), f1 (revolve->interest). "
  "Cost: f21, f14, f13, f15 (rewards + benefit give-backs). Risk: f11, f3 (collections). "
  "Relationship: f19. Dropped: id, f18 (dup of f17 r=0.93), f16 (35% at cap), f2/f12/f22/f23 "
  "(low-signal / likely decoys).",
 "Profitability Equation":
  "Prediction = rank( 0.5*rank(A) + 0.5*rank(B) ). "
  "A = (0.45*spend + 0.20*interest - 0.20*cost) * (1 - 0.50*risk) * (1 + 0.15*rel) - 0.10*collections. "
  "B = spend - 0.40*risk + 0.30*interest - 0.30*cost. "
  "spend=0.5*rank(f5)+0.5*rank(f6+..+f10); interest=rank(f1*(1-rank(f11))); "
  "cost=rank(f21+f14+50*f13+15*f15); rel=rank(f19); risk=rank(f11).",
 "Prediction Logic":
  "Continuous score in (0,1], higher = more profitable. Cardmembers rank-ordered; top 20% are the "
  "predicted most-profitable set. Continuous (not 0/1) preserves the boundary and is robust to the 70/30 split.",
 "Variable Selection Logic":
  "Features mapped to a revenue-minus-cost-minus-risk P&L; kept the economically strongest, removed "
  "identifier, duplicate lend line, near-constant and likely-decoy features. Missingness is structural "
  "(blocks {f6-f10},{f4,f21},{f17},{f23}) and imputed to 0 (non-user).",
 "Coefficient/Weight Derivation":
  "Business/AHP-set from card unit-economics, NOT fitted (no target exists). Spend is primary (alpha), "
  "interest secondary (charge card, 53% never revolve), risk multiplicative so it cannot be out-spent. "
  "Ensemble of two functional forms reduces private-leaderboard variance.",
 "Feature Transformations":
  "All economic terms percentile-rank transformed to [0,1] (immune to ~2.6% winsorization caps and skew "
  "1.4-2.8). Negative Other Spend (f7) floored at 0. Revolve risk-attenuated multiplicatively by (1-rank(f11)).",
 "Business Logic":
  "Issuer profit = interchange on spend + interest on revolving balances - reward redemptions - lifestyle "
  "credit give-backs - collections/servicing, discounted by credit risk. Deep relationships add stickiness.",
 "Assumptions":
  "UNVERIFIABLE WITHOUT LABELS: (1) which spend metric the hidden target uses - f5 and f6-f10 are "
  "orthogonal (rank corr ~0.01), so we blend 50/50; (2) the coefficient magnitudes; (3) that risk enters "
  "profit beyond what spend already implies. VERIFIED IN DATA: f11 higher=riskier (r=+0.45 w/ collections); "
  "f17~f18 duplicate (r=0.93). No demographics/tenure in data (f4~tenure, f17~affluence proxies).",
 "Validation Approach":
  "Unsupervised, parameter-free: (A) philosophy agreement - risk/revenue/unit-econ converge (top-20% "
  "Jaccard 0.53-0.57); (B) assumption sensitivity - top-20% 73-97% stable under +/-30-50% coefficient/rate "
  "shocks; (C) face validity - top-20% show ~2.8x non-travel spend, ~1.9x revolve, ~0.1x risk, ~0 collections.",
 "Additional Notes (Optional)":
  "Final model is an ENSEMBLE (unit-economics gate + lean additive), chosen for robustness on the hidden "
  "30%. Pipeline is deterministic and scales unchanged to all 500K unique_identifiers.",
}


def fill_template(template_path, ids, raw, out_path):
    import openpyxl
    # --- align raw scores to the template's exact ID order (via pandas) ---
    tmpl_ids = pd.read_excel(template_path, sheet_name="Predictions")["ID"].astype(np.int64)
    lookup = dict(zip(ids.astype(np.int64), raw.astype(float)))
    aligned = tmpl_ids.map(lookup)
    missing = int(aligned.isna().sum())
    filled = len(aligned) - missing

    # unscored IDs -> a strictly-ordered band BELOW all real scores (distinct,
    # deterministic, conservatively non-top). No effect when data covers all IDs.
    if missing:
        floor = float(np.nanmin(aligned.values)) - 1.0
        miss_pos = np.where(aligned.isna().values)[0]
        aligned.iloc[miss_pos] = floor - 1e-9 * np.arange(1, missing + 1)

    prediction = aligned.rank(pct=True)           # clean (0,1] over the FULL template

    # --- write Prediction column + Framework sheet, preserving template ---
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Predictions"]
    header = [ws.cell(row=1, column=j).value for j in range(1, ws.max_column + 1)]
    pred_col = header.index("Prediction") + 1
    vals = prediction.to_numpy()
    for i, v in enumerate(vals):                  # row i+2 aligns with tmpl order
        ws.cell(row=i + 2, column=pred_col).value = float(v)

    fw = wb["Profitability Framework"]
    for row in fw.iter_rows(min_row=2, max_row=fw.max_row):
        if row[0].value in FRAMEWORK:
            row[1].value = FRAMEWORK[row[0].value]

    wb.save(out_path)
    return len(tmpl_ids), filled, missing


# ============================================================ validation =====
def validate(template_path, out_path):
    import openpyxl
    print("\n" + "=" * 60 + "\nVALIDATION REPORT\n" + "=" * 60)
    ok = True

    tmpl = pd.read_excel(template_path, sheet_name="Predictions")
    sub = pd.read_excel(out_path, sheet_name="Predictions")

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    check("no missing Prediction values", sub["Prediction"].notna().all(),
          f"{int(sub['Prediction'].isna().sum())} nulls")
    check("row count matches template", len(sub) == len(tmpl),
          f"{len(sub)} vs {len(tmpl)}")
    check("IDs match template exactly (order+value)",
          sub["ID"].reset_index(drop=True).equals(tmpl["ID"].reset_index(drop=True)))
    check("no duplicate IDs", sub["ID"].is_unique,
          f"{int(sub['ID'].duplicated().sum())} dups")
    check("Prediction numeric in [0,1]",
          sub["Prediction"].between(0, 1).all(),
          f"range [{sub['Prediction'].min():.3f}, {sub['Prediction'].max():.3f}]")
    check("Prediction rank-ordered (near-unique, continuous)",
          sub["Prediction"].nunique() > 0.5 * len(sub),
          f"{sub['Prediction'].nunique():,} distinct / {len(sub):,}")
    check("exactly 2 sheets present",
          set(openpyxl.load_workbook(out_path).sheetnames) ==
          {"Predictions", "Profitability Framework"})

    n_top = int((sub["Prediction"] >= sub["Prediction"].quantile(0.8)).sum())
    print(f"  [INFO] top-20% flagged: {n_top:,} ({100*n_top/len(sub):.1f}%)")
    print("=" * 60)
    print("OVERALL:", "PASS ✓" if ok else "FAIL ✗")
    print("=" * 60)
    return ok


# ============================================================ main ===========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="official competition dataset (csv)")
    ap.add_argument("--template", required=True, help="official submission template xlsx")
    ap.add_argument("--out", default="submission_final.xlsx")
    args = ap.parse_args()

    print(f"Loading data: {args.data}")
    df = load_data(args.data)
    print(f"  {len(df):,} rows")

    X, flags = clean(df)
    raw = final_score_raw(X)
    print(f"Scored {len(raw):,} cardmembers | raw score range "
          f"[{raw.min():.4f}, {raw.max():.4f}]")

    n_rows, filled, missing = fill_template(args.template, df["id"], raw, args.out)
    print(f"Template rows: {n_rows:,} | filled from data: {filled:,} | "
          f"fallback (ID not in data): {missing:,}")
    if missing:
        print(f"  ⚠  {missing:,} template IDs were NOT in the supplied data and got a "
              f"conservative 0.0.\n     Re-run on the FULL official dataset so every ID is scored.")

    ok = validate(args.template, args.out)
    print(f"\nWrote {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
