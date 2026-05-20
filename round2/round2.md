# Round 2: Market Access Bidding & Resource Allocation Optimization

> **Theme:** *"Limited Market Access"* (Algorithmic) · *"Invest & Expand"* (Manual)
> **Algorithmic:** Bid for 25% additional order book quote access via a one-time Market Access Fee (MAF)
> **Manual:** Allocate a 50,000 XIRECS budget across Research, Scale, and Speed to maximize P&L

---

## Overview

Round 2 introduced game-theoretic decision-making on both fronts. The algorithmic challenge was not a trading strategy problem — it was an optimal bidding problem under distributional uncertainty. The manual challenge was a resource allocation problem where one dimension (Speed) was rank-based, making the optimization inherently dependent on modeling other participants' behavior.

Both problems shared the same core structure: **maximize expected payoff given uncertainty about opponents' actions**, with the added constraint that overbidding is costly and underbidding forfeits access entirely.

---

## Part 1: Algorithmic — Market Access Fee Bidding

### Problem Structure

The MAF is an all-or-nothing auction: submit a bid, and if it lands in the top 50% across all participants, access is granted and the fee is paid. If not, no fee is paid and no access is granted.

The expected value of market access — estimated from the gap between the official IMC backtester (running on a random 80% of quotes) and a custom backtester (running on full quote flow) — was approximately **8,000 XIRECS**.

### Bid Distribution Modeling

With no direct information on competitors' bids, the bid distribution was modeled as a **Beta distribution** on the interval [0, 9,000]:

$$
\text{Bid} \sim \text{Beta}(\alpha=2,\ \beta=5) \times 9{,}000
$$

Parameter rationale:
- Upper bound of ~9,000 anchored to Round 1 top performer's P&L as a proxy for maximum rational bid
- $\alpha=2, \beta=5$ chosen to reflect a right-skewed distribution: most participants bid conservatively, few bid aggressively
- Shape reflects **loss aversion bias** — participants willing to pay less than the full expected value to avoid the risk of overpaying

### Optimal Bid Derivation

The expected profit function was constructed as:

$$
\mathbb{E}[\text{Profit}(b)] = P(\text{win} \mid b) \times (V - b)
$$

where $V \approx 8{,}000$ is the estimated access value and $P(\text{win} \mid b)$ is the CDF of the modeled bid distribution evaluated at $b$.

Maximizing this function yielded an **optimal bid of approximately 3,580 XIRECS**.

To validate this was not simply an artifact of the expected profit curvature, a marginal cost analysis was performed: at each bid level, the cost per 1% increase in win probability was computed. The inflection point — where marginal cost begins rising sharply relative to marginal win probability gain — confirmed the ~3,580 range as the efficient frontier of the bid curve.

### Game-Theoretic Framing

This problem has the structure of a **Keynesian Beauty Contest**: the optimal bid depends not on the intrinsic value of access, but on where other participants expect each other to bid. Nash equilibrium exists where no participant can unilaterally improve their expected payoff by changing their bid.

The strategy here deliberately did not target the Nash equilibrium directly. Instead, it targeted the **region just below where marginal cost accelerates** — capturing access probability efficiently without overpaying into the tail of the distribution. The reasoning: loss-averse participants create a natural floor in the bid distribution, compressing the lower tail. Bidding just above that floor captures a disproportionate share of win probability per unit of fee paid.

**Result: bid accepted.** Market access was granted, confirming the bid landed in the top 50% of submissions.

---

## Part 2: Manual — Three-Pillar Resource Allocation

### Problem Structure

Budget: **50,000 XIRECS**, allocated across three pillars:

$$
\text{P\&L} = \text{Research}(x_R) \times \text{Scale}(x_S) \times \text{Speed}(x_{Sp}) - \text{Budget Used}
$$

| Pillar | Growth Function | Notes |
|--------|----------------|-------|
| **Research** | $200{,}000 \cdot \frac{\ln(1+x)}{\ln(101)}$ | Logarithmic; diminishing returns |
| **Scale** | Linear: 0 → 7 over [0, 100] | Constant marginal return |
| **Speed** | Rank-based: 0.1 → 0.9 | Depends entirely on competitors' allocations |

Research and Scale are deterministic given budget allocation. Speed is rank-based — making it a function of the entire field's behavior, not just one's own input.

### Speed Distribution Modeling

The key modeling decision was estimating the distribution of competitors' Speed allocations. The Speed multiplier floor (0.1 even at zero investment) creates an incentive to underinvest in Speed, while the ceiling (0.9 for the top investor) creates an incentive to overinvest. Extreme allocations in either direction were expected to be rare.

Speed allocation proportion $Y$ was modeled as:

$$
Y \sim \text{Beta}(\alpha,\ \beta)
$$

Parameters were set to reflect moderate concentration in the low-to-mid range, with a long right tail for aggressive Speed investors. LLM-suggested parameters ($\alpha=1.5, \beta=5$) were used as a starting point and stress-tested across a parameter grid.

### Sensitivity Analysis & Optimization

With Speed distribution modeled, P&L was optimized over budget allocations $(x_R, x_S, x_{Sp})$ subject to $x_R + x_S + x_{Sp} = 50{,}000$.

**Target function:**

$$
\text{Target}(\alpha, \beta) = \frac{\text{P\&L}^*(\alpha, \beta)}{\alpha_{\text{risk}} \times \beta_{\text{risk}}}
$$

where each risk term measures sensitivity of the optimal P&L to a 0.1-unit perturbation in the corresponding Beta parameter. This penalizes allocations that are highly sensitive to misspecification of the Speed distribution — the dominant source of model risk.

**Key finding from sensitivity heatmap:**
P&L was materially more sensitive to $\beta$ than $\alpha$. This asymmetry arises from the structure of the Speed payoff: the floor (0.1 multiplier) is guaranteed, but the ceiling (0.9) is competitive. The right tail of the opponent distribution — captured primarily by $\beta$ — determines how hard it is to win the Speed rank.

**Directional implication:**
As Speed investment weight increased, sensitivity to both $\alpha$ and $\beta$ stabilized. The risk-adjusted target function therefore pushed toward **higher Speed allocation** — not because Speed had the highest raw expected return, but because it reduced model risk exposure most efficiently.

### Result

**Final allocation:** Research 16% (8,000) · Scale 50% (25,000) · Speed 34% (17,000)

| Metric | Value |
|--------|-------|
| Strategy XIRECS (Research output) | 122,780 |
| Scale multiplier | ×3.5 |
| Speed hit rate | 0.54 (Rank #1,913) |
| Gross output | 233,999 |
| Budget cost | −50,000 |
| **Manual Trading P&L** | **183,999** |

**Post-mortem: the actual Speed distribution.**
After the round, the competition revealed the field's Speed investment distribution. The realized distribution was heavily zero-inflated — approximately 450 teams allocated nothing to Speed — with a secondary cluster between 10 and 40, and a long, thin tail beyond 50. This was structurally different from the Beta model used: the Beta distribution smoothed over the spike at zero and underestimated just how many teams ignored Speed entirely.

The practical implication: Speed rank #1,913 with an allocation of 34 still captured a 0.54 hit rate multiplier, meaningfully above the 0.1 floor. The direction of the optimization (elevated Speed allocation relative to Research) was correct. But the model systematically underestimated the zero-investment cluster, which compressed the true rank benefit of moderate Speed investment — the actual payoff to Speed was higher than modeled, because the competition was thinner than the Beta prior suggested.

This is a concrete example of **prior misspecification**: the model captured the right qualitative shape (most teams underinvest in Speed) but missed the point-mass at zero, which is the dominant feature of the true distribution.

---

## Key Takeaways

- **Optimal bidding is not about expected value alone.** The MAF problem required balancing access value, win probability, and marginal cost of additional probability. The efficient region sits below where marginal cost accelerates — not at the median, and not at the Nash equilibrium.
- **Rank-based payoffs change the optimization problem fundamentally.** When one dimension depends on competitors' behavior, modeling the field's distribution becomes as important as the allocation math itself.
- **Model risk should enter the objective function, not just the constraints.** Penalizing sensitivity to distributional assumptions directly in the target function changed the direction of the optimal solution. Risk-adjusted optimization and raw optimization pointed in different directions.
- **Loss aversion creates exploitable structure in auction problems.** Participants' tendency to underbid relative to expected value compresses the lower tail of the bid distribution. Targeting just above that floor captures win probability efficiently.
- **Continuous distributions miss point masses.** The Beta model correctly predicted that most participants would underinvest in Speed, but failed to capture the zero-inflation spike — roughly 450 teams allocated nothing at all. Smooth parametric priors systematically underweight degenerate behavior. When modeling strategic opponents, always consider whether a mixed strategy (some participants opting out entirely) is rational before committing to a unimodal distribution.

---

*[← Round 1: Alpha Generation & Toxic Flow Identification](../round1/round1.md)*
*[Round 3: Options Volatility Modeling & Market Making →](../round3/round3.md)*