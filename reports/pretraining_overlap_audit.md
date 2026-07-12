# Pretraining-overlap audit

Date: 2026-07-12

This audit asks whether Chronos-2 or TimesFM-2.5 could have memorized the exact
GitHub-star histories used in this study. It distinguishes **direct entity-level
contamination** from the weaker fact that model training and evaluation dates
overlap.

## Chronos-2

Official sources say Chronos-2 was released on 2025-10-20 and trained on:

- selected Chronos datasets;
- selected GIFT-Eval Pretrain datasets; and
- synthetic univariate and multivariate data.

The technical report's Table 6 lists the real univariate datasets used for
pretraining: electricity, KDD Cup, M4, Mexico City bikes, pedestrian counts,
solar, taxi, Uber TLC, USHCN, Weatherbench, Wiki pageviews, wind farms,
temperature/rain, London smart meters, cloud traces, transport, and building
energy. No GitHub-star or repository-attention dataset is listed. Wiki
pageviews are structurally related web-attention data, so broad pattern transfer
is expected and is part of the zero-shot capability being tested; it is not
entity-level memorization.

Sources:

- Model card and training-data statement:
  <https://huggingface.co/amazon/chronos-2>
- Technical report, especially Appendix A / Table 6:
  <https://arxiv.org/abs/2510.15821>
- Official release announcement (2025-10-20):
  <https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting>

## TimesFM-2.5

The official model-card history states that TimesFM-2.5 uses:

- a subset of GIFT-Eval Pretrain;
- Wikimedia Pageviews, with a November 2023 cutoff;
- Google Trends top queries, with an end-of-2022 cutoff; and
- synthetic and augmented data.

The published GIFT-Eval Pretrain catalog spans energy, transport, nature,
cloud operations, finance, and related domains; no GitHub-star dataset was
identified. The known web-attention sources predate the breakout cohort's
2024--2026 evaluation period.

Sources:

- Official repository:
  <https://github.com/google-research/timesfm>
- Model card / training-data history:
  <https://huggingface.co/google/timesfm-2.5-200m-pytorch>
- GIFT-Eval Pretrain catalog:
  <https://huggingface.co/datasets/Salesforce/GiftEvalPretrain>

## Verdict

**No evidence of direct GitHub-star memorization was found.** The disclosed
corpora do not list GitHub stars, and TimesFM's known web-attention sources have
published cutoffs before the breakout evaluation period. This reduces, but does
not eliminate, contamination risk:

1. The complete item-level contents and timestamp cutoffs of every selected
   Chronos/GIFT series are not independently auditable from the model cards.
2. Both models were released after some or all evaluated observations already
   existed publicly.
3. None of the 179 repositories eligible under the study's 270-day history rule
   was created after either model release. The 25--29 post-release repositories
   in the ingestion snapshot are too young for this backtest.

Therefore the current result should be described as **zero-shot on a novel
dataset with no known direct training overlap**, not as a guaranteed
post-training temporal holdout.

## Clean control

The strongest control is prospective:

- freeze the current model weights and evaluation protocol;
- collect repositories created after 2025-10-20;
- evaluate windows whose complete context and target occur after that date; and
- preregister cohort construction and metrics before outcomes are visible.

That control cannot be created retrospectively from the current 270-day-eligible
sample. It is the highest-priority extension for a venue submission.
