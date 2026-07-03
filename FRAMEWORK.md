# Premier Card Profitability — Round 1 Final Submission

Rank-orders premier cardmembers by estimated profitability and flags the top 20%.
No target variable exists, so the score is a **business-driven, rank-normalized
unit-economics ensemble**, validated by philosophy-agreement and
assumption-sensitivity rather than by fitting.

## Final model — ENSEMBLE (rank-average of two forms)

Chosen over any single model because two functional forms that agree on the
profitable core but fail differently reduce variance on the hidden 30% private
leaderboard.

```
spend    = 0.5·rank(f5) + 0.5·rank(f6+f7+f8+f9+f10)      # f5 ⟂ categories → blend hedge
risk     = rank(f11)
interest = rank(f1 · (1 − risk))                         # risk-adjusted revolve interest
cost     = rank(f21 + f14 + 50·f13 + 15·f15)             # rewards + benefit give-backs
rel      = rank(f19)

# Model A — unit-economics, multiplicative risk gate
A = (0.45·spend + 0.20·interest − 0.20·cost) · (1 − 0.50·risk) · (1 + 0.15·rel) − 0.10·f3

# Model B — lean additive spend model
B = spend − 0.40·risk + 0.30·interest − 0.30·cost

Prediction = rank( 0.5·rank(A) + 0.5·rank(B) )           # (0,1], higher = more profitable
```

### Coefficients
| Coef | Value | Meaning |
|---|---|---|
| α 0.45 | spend weight — interchange is the dominant charge-card revenue |
| β 0.20 | risk-adjusted interest — real but secondary (53% never revolve) |
| γ 0.20 | variable cost (rewards + benefits) |
| λ 0.50 | multiplicative risk penalty — cannot be out-spent |
| ρ 0.15 | relationship-depth bonus (stickiness/LTV) |
| δ 0.10 | collections hard penalty (distress) |
| a,b,c | 0.40/0.30/0.30 — Model B additive weights |

## Transformations & scaling
- Percentile-rank every input → [0,1]; immune to the ~2.6% winsorization caps and skew (1.4–2.8).
- Missing → 0 (genuine non-user) + informative flags for blocks {f6–f10},{f4,f21},{f17},{f23}.
- Negative Other Spend (f7) floored at 0.

## Assumptions that CANNOT be verified without labels
1. **Which spend metric the target uses** — f5 and Σ(f6–f10) are orthogonal (rank corr ≈ 0.01,
   top-20% overlap 0.15). We blend 50/50; resolve on the public leaderboard if submissions allow.
2. **Coefficient magnitudes** — expert/AHP-set from card economics, not fitted.
3. **That risk enters profit beyond spend** — f11 is already −0.38 correlated with spend.

Verified in data: f11 higher = riskier (r=+0.45 w/ collections); f17≈f18 (r=0.93, dropped one).
No demographics/tenure in data (f4 ≈ tenure proxy, f17 ≈ affluence proxy).

## Validation (unsupervised, parameter-free)
- **Philosophy agreement:** risk/revenue/unit-econ converge (top-20% Jaccard 0.53–0.57).
- **Assumption sensitivity:** top-20% is 73–97% stable under ±30–50% coefficient/rate shocks.
- **Face validity:** top-20% show ~2.8× non-travel spend, ~1.9× revolve, ~0.1× risk, ~0 collections.

## Reproduce
```
pip install -r requirements.txt
python final_submission.py \
    --data <official_500k_dataset.csv> \
    --template 6a3cb64c7cae4_campus_challenge_r1_submission_template.xlsx \
    --out submission_final.xlsx
```
Deterministic; runs unchanged on all 500K `unique_identifiers`. Prints a validation
report and writes `submission_final.xlsx` (Predictions + filled Profitability Framework).

## Files
`final_submission.py` (submission generator) · `score_lean.py` (leaderboard probes) ·
`score_premier.py` (S1–S5 philosophies) · `eda_premier.py` (EDA) · `stress_test.py` ·
`amex_r1_profitability.ipynb` (production notebook) · `requirements.txt`.
