"""Stress-test the FINAL profitability score against real population distributions."""
import numpy as np, pandas as pd
df = pd.read_csv("book1_clean.csv")
FEATS = [f"f{i}" for i in range(1,24)]
X = df[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
X["f7"] = X["f7"].clip(lower=0)

# empirical rank map: value -> population percentile, per feature
def pct_of(col, val):
    return (X[col] <= val).mean()

CAT = ["f6","f7","f8","f9","f10"]
cat_sum_pop = X[CAT].sum(axis=1)

# ---- FINAL SCORE (coefficients documented in the writeup) ----
A = dict(alpha=0.45, beta=0.20, gamma=0.20, lam=0.5, rho=0.15, delta=0.10)

def spend_rank(row):
    f5r = pct_of("f5", row["f5"])
    catr = (cat_sum_pop <= sum(row[c] for c in CAT)).mean()
    return 0.5*f5r + 0.5*catr

def score_raw(row):
    SP   = spend_rank(row)
    risk = pct_of("f11", row["f11"])
    intr = (X["f1"]*(1-X["f11"].rank(pct=True)) <= row["f1"]*(1-risk)).mean()
    cost = ((X["f21"]+X["f14"]+50*X["f13"]+15*X["f15"]) <=
            (row["f21"]+row["f14"]+50*row["f13"]+15*row["f15"])).mean()
    rel  = pct_of("f19", row["f19"])   # relationship proxy
    base = A["alpha"]*SP + A["beta"]*intr - A["gamma"]*cost
    s = base * (1 - A["lam"]*risk) * (1 + A["rho"]*rel) - A["delta"]*row["f3"]
    return s

# population score distribution -> to convert a raw score into a percentile
def build_pop_scores():
    SP = 0.5*X["f5"].rank(pct=True) + 0.5*cat_sum_pop.rank(pct=True)
    risk = X["f11"].rank(pct=True)
    intr = (X["f1"]*(1-risk)).rank(pct=True)
    cost = (X["f21"]+X["f14"]+50*X["f13"]+15*X["f15"]).rank(pct=True)
    rel  = X["f19"].rank(pct=True)
    base = A["alpha"]*SP + A["beta"]*intr - A["gamma"]*cost
    return base*(1-A["lam"]*risk)*(1+A["rho"]*rel) - A["delta"]*X["f3"]
POP = build_pop_scores()
def to_percentile(s): return (POP <= s).mean()

# baseline = a typical active customer (medians of NON-zero where meaningful)
base = {c: X[c].median() for c in FEATS}
base.update(f5=X["f5"].median(), f11=X["f11"].median())
# make baseline a real spender so ratios are meaningful
for c in CAT: base[c] = X[c][X[c]>0].median()
base["f1"] = X["f1"][X["f1"]>0].median()
base["f21"] = X["f21"][X["f21"]>0].median()

scen = {
 "BASELINE": {},
 "Spend x2 (all categories)": {c:base[c]*2 for c in CAT} | {"f5":base["f5"]*2},
 "Risk x2":                    {"f11":base["f11"]*2},
 "Reward redemption x2":       {"f21":base["f21"]*2},
 "Travel spend x2 (f6,f9)":    {"f6":base["f6"]*2,"f9":base["f9"]*2},
 "Credit utilization x2 (f1)": {"f1":base["f1"]*2},
 "Logins -50% (f12)":          {"f12":base["f12"]*0.5},
 "Collections flag on (f3=1)": {"f3":1},
}
b0 = to_percentile(score_raw(base))
print(f"{'Scenario':<30}{'score pct':>10}{'Δ vs base':>12}")
print("-"*52)
for name, delta in scen.items():
    row = dict(base); row.update(delta)
    p = to_percentile(score_raw(row))
    print(f"{name:<30}{p*100:>9.1f}%{(p-b0)*100:>+11.1f}")
