# Round 5: Pairs Trading — Post-Mortem

> **Theme:** Pairs trading across a 40+ product universe spanning 10 sector groups
> **Outcome:** Strategy collapsed in live environment. The most instructive round of the competition.

---

## Overview

Round 5 introduced the largest and most complex asset universe of the competition — 40+ products across 10 sector groups (GALAXY_SOUNDS, MICROCHIP, SNACKPACK, SLEEP_POD, TRANSLATOR, and others). The organizers hinted at a **lead-lag relationship** between assets as the key to alpha generation.

That hint became the problem.

This document is not a strategy writeup. It is a forensic account of how a weak in-sample signal — never validated out-of-sample — made it into live deployment, and what the failure cost.

The full exploratory research process is documented in [`research_code.ipynb`](./_source_code/research_code.ipynb).

---

## The Search Process

### Phase 1: Universe Screening

The first pass covered the full product universe: spread distributions, OBI (Order Book Imbalance) statistics, return correlation matrices, and cross-group correlation structure.

**What the data showed:**
- Intra-group correlations were high and positive — products within the same sector moved together
- Cross-group correlations were close to zero — no obvious cross-sector pair candidates
- OBI had weak forward predictive power across most products — correlations between OBI at time *t* and returns at *t+1* were small and inconsistent across days

This ruled out most obvious pair structures early. The search narrowed to intra-sector lead-lag.

### Phase 2: Sector-Level Lead-Lag

Sector indices (price sums per group) were visualized day by day to test whether any sector consistently moved before another.

**Finding:** No consistent cross-sector timing pattern. Sector indices moved roughly contemporaneously. The aggregation likely masked intra-sector structure, but no sector-level signal survived.

### Phase 3: Intra-Sector OBI Lead-Lag

The main research effort. For each sector, OBI spikes in one product were used as signals, and the forward price response of other products in the same sector was measured over 1–40 tick horizons.

Sectors tested: **SNACKPACK**, **TRANSLATOR**, **SLEEP_POD**.

| Sector | Pair Tested | Finding |
|--------|------------|---------|
| SNACKPACK | VANI/RASP/STRAW OBI → CHOCO price | Signal smaller than spread cost. Negative net edge. |
| SNACKPACK | VANI/RASP/STRAW OBI → PISTACHIO price | Weak correlation at short lags, no persistence. |
| SNACKPACK | PCA min-vol basket | Variance reduced, no mean-reverting residual. |
| TRANSLATOR | SPACE_GRAY/CHARCOAL/VOID OBI → GRAPHITE price | OBI near zero, forward curves flat. Noise. |
| SLEEP_POD | POLYESTER OBI → NYLON price | Weak directional response at 1–10 ticks on Day 2. |
| SLEEP_POD | NYLON OBI → POLYESTER price | No consistent response. |

---

## The Critical Failure Point

SLEEP_POD POLYESTER → NYLON was the only candidate that showed any directional structure. On Day 2, buy signals (OBI ≥ 0.5 on POLYESTER) produced a slight positive forward return in NYLON over 5–10 ticks.

This is where the process broke down.

**What should have happened:**
1. Validate the signal on Day 3 and Day 4 before treating it as real
2. Test whether the edge survived actual transaction costs (entry at ask, exit at bid)
3. Run cointegration testing on the POLYESTER/NYLON spread before constructing a Z-score

**What actually happened:**
The organizer's lead-lag hint made the Day 2 observation feel like confirmation rather than a hypothesis. Under time pressure, the team combined this weak signal with ad-hoc Z-score pair construction — no cointegration test, no OOS validation — and deployed it.

The extended lag analysis (40 ticks, decomposed by actual entry prices) made the problem clear in hindsight: even at 40 ticks, the profit zone between entry price and future bid/ask was minimal to negative. The signal that appeared directional at mid-price on Day 2 did not survive spread costs, and it did not generalize to other days.

**In the live environment, the strategy collapsed.** P&L and final standings both took the hit.

---

## Forensic Diagnosis

**Why did a Day 2 signal make it to deployment?**

Three compounding failures:

**1. Anchoring bias.** The organizer's hint created a prior that a lead-lag signal *existed*. Every weak pattern that fit the narrative felt like evidence. Every pattern that didn't fit was set aside. The prior should have been skeptical by default — "show me this survives OOS" — not confirmatory.

**2. Time pressure collapsed the validation standard.** Rigorous OOS testing takes time. Under competition pressure, the implicit threshold for "good enough to deploy" dropped from "statistically robust" to "looked right on the chart." These are not the same thing.

**3. Z-score without cointegration is curve fitting.** Constructing a Z-score spread between two assets assumes the spread is stationary. That assumption needs to be tested — Engle-Granger, Johansen, or at minimum an ADF test on the spread series. None of this was done. The strategy was built on an assumption that was never verified.

---

## What This Round Changed

The Round 5 failure reoriented the research philosophy more than any other round.

The specific changes:

**On signal validation:** Any signal that appears on a single day is a hypothesis, not a finding. Minimum bar for deployment is now OOS validation across at least two held-out periods before any live execution.

**On external hints:** Competition organizer nudges are priors, not facts. They narrow the search space — they don't replace the statistical discipline required to validate what you find in that space.

**On pairs construction:** Z-score spread trading requires verified cointegration. Without it, the strategy is in-sample curve fitting with a quantitative-looking wrapper.

**On time pressure:** The moment time pressure starts compressing validation standards, that is the signal to *slow down*, not speed up. A bad strategy deployed is worse than no strategy at all — it actively destroys P&L.

---

## Key Takeaways

- **A Day 2 signal is a hypothesis.** The only question that matters is whether it survives Day 3 and Day 4. If there's no time to check, there's no time to deploy.
- **Anchoring bias is most dangerous when the anchor feels authoritative.** An organizer hint, a confident teammate, a clean-looking chart — any of these can suppress the skepticism that rigorous research requires.
- **Cointegration is not optional in pairs trading.** OBI lead-lag and Z-score construction are both downstream of a stationarity assumption. If the spread isn't stationary, everything built on top of it is fiction.
- **The cost of a bad live strategy exceeds the cost of sitting out.** Deploying an unvalidated strategy doesn't just miss alpha — it generates negative P&L and compounds the damage done by the missed opportunity cost.

---

*[← Round 4: Trader Microstructure Analysis & Exotic Options Portfolio](../round4/ROUND4_DESCRIPTION.md)*