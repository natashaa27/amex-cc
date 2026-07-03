# Premier Card Profitability — Round 1 Framework

Rank-orders premier cardmembers by estimated profitability to the issuer and
flags the top 20%. No target variable exists, so the score is a **business-driven
unit-economics equation**, rank-normalized and validated by philosophy-agreement
and assumption-sensitivity rather than by fitting.

## Pipeline

| Script | Purpose |
|--------|---------|
| `eda_premier.py` | Full EDA: distributions, missingness, outliers, skew, correlation, redundancy checks, unsupervised importance. |
| `score_premier.py` | Builds S1–S5 scores, emits ranked `Prediction`, runs validation (A) philosophy agreement + (B) assumption sensitivity. |

Run:
```
python eda_premier.py   --data <data.csv> --out eda_out
python score_premier.py --data <data.csv> --out scores_out   # -> scores_out/predictions.csv
```

## Key data facts (from EDA on 274K rows)

- **Winsorized/synthetic data:** ~2.6% of every monetary feature sits at an identical
  cap (e.g. f7=146700.554, f1=17967.726). f16 has 35% at its cap (low signal).
- **Missingness is 100% structural**, in blocks: {f6–f10} (23%), {f4,f21} (51%),
  {f17} (58%), {f23} (88%). Encoded as flags, imputed to 0.
- **f5 is NOT the category-spend total** (1.3% match) — different scale; categories carry spend.
- **Redundancy:** f17~f18 = 0.93 (drop one); spend block f6–f10 = 0.64–0.80 (one factor).
- **Validated interactions:** f1~f11 = +0.58 (revolve×risk), f11~f3 = +0.45 (risk×collections)
  → confirms higher f11 = riskier; revolve revenue must be risk-attenuated.

## Scoring philosophies

| Score | Philosophy | Idea |
|-------|-----------|------|
| S1 | Conservative Banking | Loss-avoidance dominates |
| S2 | Revenue Maximization | Interchange + interest top-line |
| S3 | Long-term Value | Relationship depth + tenure + retention |
| S4 | Premium Card Economics | Faithful revenue − cost − expected-loss P&L |
| **S5** | **Balanced (submitted)** | Mean of percentile-ranks of S1–S4 |

**S4 equation (dollar unit-economics):**
```
Profit = Interchange + Interest − Rewards − Benefits − Servicing − ExpectedLoss
Interchange   = 0.018 · (1.0·f7 + 0.9·f10 + 0.8·f8 + 0.6·f6 + 0.6·f9)
Interest      = 0.18 · f1
Rewards       = 0.010 · f21
Benefits      = f14 + f16 + 50·f13 + 15·f15
Servicing     = 20·f2 + 100·f3
ExpectedLoss  = f11 · (f1 + 0.1·f17) · 0.90
```

## Validation (unsupervised, parameter-free)

- **(A) Philosophy agreement:** risk/revenue/unit-econ converge (top-20% Jaccard 0.53–0.57);
  LTV deliberately diverges (0.22–0.35). S5 sits in the consensus (Spearman 0.74–0.95).
- **(B) Assumption sensitivity:** S4 top-20% is **80–97% stable** under ±30–50% rate shocks.
- **(C) Face validity:** top-20% show 6.5× non-travel spend, 2.9× revolve, ~1/10th risk,
  ~0 collections — the correct profitable-premier profile.

## Assumptions & limits

- No demographics/tenure in data; f4 ≈ tenure proxy, f17 ≈ affluence proxy.
- Rates (interchange 1.8%, net APR 18%, $0.010/point, LGD 0.90) are industry-reasonable,
  not fitted. Annual fee is constant across premier holders → omitted (no rank effect).
- Available data is a 274K sample; `score_premier.py` runs unchanged on all 500K
  `unique_identifiers` for the final submission.
