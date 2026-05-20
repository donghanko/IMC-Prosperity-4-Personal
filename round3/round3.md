# Round 3: Options Volatility Modeling & Market Making

> **Theme:** *"Options Require Decisions"*

> **Assets:** ① HYDROGEL_PACK ②  VELVETFRUIT_EXTRACT (delta-1) ③  VELVETFRUIT_EXTRACT_VOUCHERs (options, varying 10 strikes)

> **Structure:** European-style calls, 7-day expiry from Round 1. No early exercise. Positions liquidated at hidden fair value at round end.

---

## Overview

Round 3 introduced the first options layer of the competition. With 10 strikes and no explicit label of call vs. put, the first task was purely empirical: identify option type through data visualization before any pricing model could be applied.

The round exposed a recurring tension — theoretically sound frameworks repeatedly running into the hard wall of spread costs and microstructure. Most of the meaningful learning came from understanding *why* each approach broke down.

---

## 1. Identifying Option Type & Classifying the Strike Universe

With no call/put label provided, option type was inferred from price behavior relative to the underlying. The strike universe split cleanly into three regimes:

**Deep ITM**
Price movement in the 1,200–1,300 range, consistent with intrinsic value ($S_t - K$) plus residual time value. With 7 days to expiry, time value approached zero. Delta near 1 — effectively a leveraged delta-1 product with no meaningful edge unless arbitrage existed.

**ATM**
The highest-edge segment. Delta fluctuates most sharply here, creating the largest potential for mispricing. Time value of ~50 remained even at 7 days — tradable premium still on the table.

**Deep OTM**
Mid-price quoted at 0.5, but actual trades executed at 0. Functionally illiquid. Only use case: holding as a tail hedge against a sharp underlying rally.

---

## 2. Arbitrage Screening

Option price = intrinsic value + premium. Prices cannot fall below intrinsic value — any breach is an arbitrage.

Arbitrage opportunities, when they appeared, were concentrated in deep ITM/OTM where premium compression is greatest. OTM was effectively untradeable, so the screen focused on ITM.

**The spread cost problem.**
Mid-price signals showed 5–10 ticks of edge per signal. Actual spreads exceeded 30 ticks. Taking liquidity at those spreads meant a realized loss of 20–25 per entry — erasing any theoretical edge entirely.

The implication was structural: **any profitable strategy in this options market had to be a maker, not a taker.**

> Deep ITM options (strikes 4000/4500) → market making with passive quotes at mid ± 2 ticks.
> Fair value as an anchor made this viable.

Calendar arbitrage was screened separately. Convexity held throughout — no calendar arb opportunities were found.

---

---

## 3. Gamma Scalping

**The idea.**
Delta-hedge an ATM option (long option, short $\Delta$ units of underlying). As the underlying moves and delta shifts, rebalance — selling into rallies, buying into dips. The accumulated rebalancing P&L converges to the gamma — the gap between the option's price curve and its delta tangent.

**The condition for profitability.**
Gamma income must exceed theta decay and transaction costs. 

In this specific simulation setup, **the deterministic theta decay parameter was structurally omitted ($\Theta \approx 0$)** because the option chain lacked cross-sectional expiration variance (i.e., multiple maturities per identical strike did not coexist). Since time value did not decay asynchronously across different horizons, the options trading environment theoretically heavily favored a long-gamma position.

**Why it failed.**
Despite the structural absence of theta drag, rebalancing required crossing the wide spread on the underlying delta-1 asset. The gamma realized per rebalancing cycle was mathematically insufficient to absorb those heavy transaction costs. The underlying asset simply didn't exhibit enough realized volatility relative to the strike spacing to generate gamma income that outpaced the systematic spread drag.

This friction was only fully decoded after computing the Implied Volatility (IV) dynamics across the entire cross-sectional dataset—covered in the next section.

---

## 4. Implied Volatility

IV was extracted via bisection method using the Black-Scholes-Merton framework:

$$
C(S, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
$$

$$
d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)(T-t)}{\sigma \sqrt{T-t}}, \quad
d_2 = d_1 - \sigma\sqrt{T-t}
$$

Risk-free rate $r$ was set to zero — with annualized TTM near zero, carry effects were negligible, and the model was being used as a signal benchmark rather than a pricing engine.

**Key findings.**

The cross-sectional IV profile showed **clustering** rather than the expected smile or skew structure. This explained the gamma scalping failure: realized moves relative to strike were too small to generate meaningful gamma income against spread costs.

The time-series of IV told a different story: most strikes exhibited a persistent **upward trend in IV as TTM decreased**. This was noted at the time but not decoded until Round 4 — where it was identified as a **volatility clustering** signature visible through log-moneyness. Catching this signal in Round 3 would have meaningfully changed the options strategy.

---

## 5. Fair Value Modeling

### Orc-Wing Fitting

For market making to work, a reliable fair value is required. The approach: fit a parametric model to the IV surface, extract parameters, and compute fair option prices continuously from log-moneyness.

The Orc-Wing model was chosen over SABR for its wing flexibility:

$$
\sigma(x) = \sigma_{ref} \cdot \left[ 1 + SSR \left( \frac{V_c \cdot (1 - \rho) \cdot f_c(x) + V_p \cdot (1 + \rho) \cdot f_p(x)}{2} \right) \right]
$$

Parameters:
- $\sigma_{ref}$: ATM volatility level
- $SSR$: slope/skew of the smile
- $V_c, V_p$: call/put wing weights
- $f_c, f_p$: call/put curvature functions
- $\rho$: skew asymmetry

*Note: "call/put" here refers to ITM side, not option type — call wing = right of ATM.*

The fitted surface was used to generate Z-scores per option. Above a threshold → sell; below → buy.

**Result: overfit.** Performance improved on specific options and specific days in a pattern that was clearly data-specific rather than structural. Cherry-picking those options was tempting — but the signal had no generalizable basis. The Orc-Wing curve was not describing the market's actual IV dynamics.

### Realized Volatility (Model-Free)

A model-free approach was also explored: extracting the market's expected total variance from the cross-section of call prices, analogous to VIX construction.

$$
\sigma = \sqrt{\frac{2}{T} \int_0^\infty \frac{C(T,K) - \max(F_0 - K, 0)}{K^2} \, dK}
$$

Discretized across available strikes, this gives a fair volatility estimate without parametric assumptions. In practice, it produced a single constant — less adaptive than Orc-Wing and with the same overfitting risk in a small strike universe. Not used in production.

---

## 6. Final Strategy & Honest Assessment

**What was deployed:** Market making on deep ITM options (passive quotes, mid ± 2 ticks) + making/taking strategies on the delta-1 futures products.

**What didn't work:** No options signal beyond ITM market making produced robust out-of-sample P&L. Multiple approaches were tested — more than documented here — but none survived the step from in-sample calibration to live execution cleanly enough to deploy.

**The signal that was missed:** The IV term structure — persistent upward slope as TTM decreased — contained a structural signal that only became legible in Round 4. It was the most consequential missed observation of the round.

---

## Key Takeaways

- **Spread costs are the first filter, not an afterthought.** Mid-price signals are theoretical. Any strategy in a wide-spread options market lives or dies by whether it can operate as a maker.
- **IV shape is a structural diagnostic.** The clustering pattern and upward-sloping term structure were present from the start — they just weren't interpreted correctly in time.
- **Parametric fitting on thin data overfits.** Orc-Wing is a powerful model; it was applied to a market with too few strikes and too much noise to generalize. Model selection has to account for data regime, not just theoretical elegance.

---

*[← Round 2: MAF Auction Bidding & Resource Allocation Optimization](../round2/round2.md)*
*[Round 4: Trader Microstructure Analysis & Exotic Options Portfolio →](../round4/round4.md)*