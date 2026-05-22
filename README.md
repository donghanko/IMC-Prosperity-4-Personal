# Y-FoRM: IMC Prosperity 4 — Individual Research Archive

> **Peak Standing: Global #200 · National (South Korea) #1** (through Round 4)

Most competition writeups document what worked. This one documents what didn't — and why that matters more.

This repository archives the full trail of my **individual research iterations** throughout IMC Prosperity 4 (2026): the model derivations, dead ends, and pivots in thinking that never make it into a polished final submission. The goal isn't to showcase clean results. It's to show how I think when the market pushes back.

> **Note on scope:** All models, derivations, and simulation plots here reflect my own research footprint. No shared team assets or production-level code are published, out of respect for team confidentiality. Finalized production strategies will be documented in a separate repository.

---

## Repository Structure

```
IMC-Prosperity-4-Personal/
├── round1/
│   ├── ROUND1_DESCRIPTION.md   # description
│   ├── _source_code/           # analysis source code with output markdown file
│   │   └── _analysis_img/      # png output images referenced in RESEARCH_CODE.md
│   ├── _source_data/           # data set
│   ├── _img/                   # png files for Description.md
│   └── ...
├── round2/
│   ├── ROUND2_DESCRIPTION.md   # description
│   ├── _source_code/           # analysis source code with output markdown file
│   │   └── _analysis_img/      # png output images referenced in RESEARCH_CODE.md
│   ├── _source_data/           # data set
│   ├── _img/                   # png files for Description.md
│   └── ...
├── round3/
│   ├── ROUND3_DESCRIPTION.md   # description
│   ├── _source_code/           # analysis source code with output markdown file
│   │   └── _analysis_img/      # png output images referenced in RESEARCH_CODE.md
│   ├── _source_data/           # data set
│   ├── _img/                   # png files for Description.md
│   └── ...
├── round4/
│   ├── ROUND4_DESCRIPTION.md   # description
│   ├── _source_code/           # analysis source code with output markdown file
│   │   └── _analysis_img/      # png output images referenced in RESEARCH_CODE.md
│   ├── _source_data/           # data set
│   ├── _img/                   # png files for Description.md
│   └── ...
├── round5/
│   ├── ROUND5_DESCRIPTION.md   # description
│   ├── _source_code/           # analysis source code with output markdown file
│   │   └── _analysis_img/      # png output images referenced in RESEARCH_CODE.md
│   ├── _source_data/           # data set
│   └── ...          
└── README.md
```



## Overview

Rounds 1–2 were **qualifier rounds (Phase 1)** and do not count toward the final standings. Rounds 3–5 were the main competition.

| Round | Theme | Methods |
|-------|-------|---------|
| Round 1 *(Qualifier)* | Market Making & Inventory Management | LOB analysis, linear trend fitting, tick distribution, toxic flow screening |
| Round 2 *(Qualifier)* | Game Theory & Resource Allocation | Beta distribution modeling, marginal cost analysis, CVaR optimization |
| Round 3 | Options Trading | Orc-Wing fitting, BSM IV extraction, gamma scalping, spread-cost analysis |
| Round 4 | Trader Identity & Microstructure | Trader taxonomy, OU process, log-moneyness, simulated annealing |
| Round 5 | Pairs Trading | OBI lead-lag analysis, cointegration testing, anchoring bias forensics |

---

## Key Research Themes

**Bridging theory and market friction.**
The core tension throughout this competition was reconciling mathematically rigorous frameworks — SABR, Orc-Wing, OU processes — with the realities of a simulated limit order book: spread costs, toxic order flow, and regime shifts that don't announce themselves. That gap is where most of the alpha was lost, and where most of the learning happened.

**Diagnosing overfitting in real time.**
Rounds 3 and 5 both involved strategies that looked strong in-sample and failed out-of-sample. Documenting the precise moment each broke down — and why — is the central purpose of this archive.

---

## Post-Mortem: Where I Got It Wrong

### Round 3–4 · The Cost of Delayed Structural Insight

In Round 3, I committed significant effort to fitting the **Orc-Wing model** to the implied volatility surface and building a **gamma scalping** strategy around it. The IV surface was persistently upward-sloping as TTM decreased — a structural signal I noted but didn't immediately decode.

It wasn't until Round 4 that I identified the mechanism: **volatility clustering**, visible through log-moneyness, was driving the short-dated IV premium. By then, the window for capturing that alpha in the options space had already closed.

**What changed:** I now treat IV surface shape as a first-pass diagnostic before committing to any model family. Structural signals don't wait for theoretical confirmation.

---

### Round 5 · Anchoring Bias and the Overfitting Trap

The final round involved pairs trading. The competition organizers hinted at a **lead-lag relationship** between assets — and that hint became the problem.

I spent the majority of my available research time attempting to capture this lag through cross-correlation modeling. Nothing produced a robust out-of-sample signal. Under time pressure, the team fell back on an ad-hoc Z-score combination that fit the in-sample P&L curve with no cointegration testing to back it.

When the strategy hit the live environment, it collapsed. The drawdown affected both P&L and final standings.

**What changed:** The failure reframed how I approach competition hints entirely. An organizer nudge is a hypothesis, not a fact — it still needs to survive rigorous OOS validation before deployment. Anchoring on the hint cost us the statistical discipline that should have been non-negotiable.

---

## Concluding Framework

This competition reinforced a single principle I now treat as foundational:

> *A model that works beautifully in-sample will decay out-of-sample the moment statistical discipline is traded for a quick fix.*

The Round 5 failure in particular shifted my research philosophy — from chasing alpha signals toward building **regime-adaptive risk guardrails** that hold even when the market structure stops cooperating.

The scars are in the writeups. The framework they produced is what this archive is really about.

## 📊 Jupyter Notebook Viewer Links

GitHub often fails to render large Jupyter notebooks. If you want to view the full analysis including outputs and graphs, please use the Google Colab links below:

**Round 1**
- [research_code.ipynb](https://colab.research.google.com/github/donghanko/IMC-Prosperity-4-Personal/blob/main/round1/_source_code/research_code.ipynb)


**Round 2**
- [research_code.ipynb](https://colab.research.google.com/github/donghanko/IMC-Prosperity-4-Personal/blob/main/round2/_source_code/research_code.ipynb)

**Round 3**
- [research_code.ipynb](https://colab.research.google.com/github/donghanko/IMC-Prosperity-4-Personal/blob/main/round3/_source_code/research_code.ipynb)

**Round 4**
- [research_code.ipynb](https://colab.research.google.com/github/donghanko/IMC-Prosperity-4-Personal/blob/main/round4/_source_code/research_code.ipynb)

**Round 5**
- [research_code.ipynb](https://colab.research.google.com/github/donghanko/IMC-Prosperity-4-Personal/blob/main/round5/_source_code/research_code.ipynb)

---

## Contents

- [Round 1: Market Making & Inventory Management](./round1/ROUND1_DESCRIPTION.md)
- [Round 2: Game Theory & Resource Allocation](./round2/ROUND2_DESCRIPTION.md)
- [Round 3: Options Volatility Modeling & Market Making](./round3/ROUND3_DESCRIPTION.md)
- [Round 4: Trader Identity & Microstructure Analysis](./round4/ROUND4_DESCRIPTION.md)
- [Round 5: Pairs Trading Post-Mortem & Overfitting Analysis](./round5/ROUND5_DESCRIPTION.md)
