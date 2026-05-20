# Round 2: MAF Auction Bid Optimization

> **Problem:** All-or-nothing auction — bid in the top 50% of all participants to gain 25% additional quote access.  
> **Access value:** ~8,000 XIRECS (estimated from gap between official backtester and full-quote-flow custom backtester).

## Overview

The MAF bid is an expected-profit optimization under distributional uncertainty:

$$
\mathbb{E}[\text{Profit}(b)] = P(\text{win} \mid b) \times (V - b)
$$

where V ≈ 8,000 and P(win | b) = CDF of the modeled bid distribution at b.

**Bid distribution model:** Beta(α=2, β=5) on [0, 9,000]  
- Upper bound anchored to Round 1 top performer P&L as proxy for maximum rational bid  
- Right-skewed: most participants bid conservatively (loss aversion)

**Optimal bid:** ~3,580 XIRECS (maximizes expected profit)  
**Result:** Bid accepted — market access granted.


---
## Section 1: Beta Distribution — Bid Density & CDF

Visualize the Beta(2, 5) distribution scaled to [0, 9,000].  
This represents the prior belief about where competitors' bids will land.



```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

# -------------------------
# Parameters
# -------------------------

alpha = 2
beta_param = 5

scale_max = 9000
bid_value = 2800

# x grid
x = np.linspace(0, scale_max, 1000)

# scaled beta pdf
y = beta.pdf(x / scale_max, alpha, beta_param) / scale_max

# -------------------------
# Percentiles
# -------------------------

median = beta.ppf(0.5, alpha, beta_param) * scale_max
p95 = beta.ppf(0.95, alpha, beta_param) * scale_max
p99 = beta.ppf(0.99, alpha, beta_param) * scale_max

percentile_2800 = beta.cdf(
    bid_value / scale_max,
    alpha,
    beta_param
)

print("Median:", median)
print("95th percentile:", p95)
print("99th percentile:", p99)
print("Percentile of 2800:", percentile_2800)

# -------------------------
# Plotly Figure
# -------------------------

fig = go.Figure()

# PDF curve
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Beta(2,5) PDF"
    )
)

# Vertical lines
def add_vline(x_val, label):
    fig.add_vline(
        x=x_val,
        line_dash="dash",
        annotation_text=label,
        annotation_position="top"
    )

add_vline(median, "Median")
add_vline(p95, "95th %")
add_vline(bid_value, "Bid=2800")

# Layout
fig.update_layout(
    title="Scaled Beta(2,5) Distribution on [0, 9000]",
    xaxis_title="Payoff",
    yaxis_title="Density",
    template="plotly_white",
    width=900,
    height=500
)

fig.show()
```

    Median: 2380.0498496609403
    95th percentile: 5236.230683268233
    99th percentile: 6351.176954877367
    Percentile of 2800: 0.6035109233950711
    




```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

# -------------------------
# Parameters
# -------------------------

alpha = 2
beta_param = 5

scale_max = 9000
profit_gain = 8000

# bid grid
bids = np.linspace(1000, 5000, 400)

expected_profit = []

for b in bids:

    # entry probability
    p_enter = beta.cdf(
        b / scale_max,
        alpha,
        beta_param
    )

    # correct formula
    exp_p = p_enter * (profit_gain - b)

    expected_profit.append(exp_p)

# optimal bid
best_idx = np.argmax(expected_profit)
best_bid = bids[best_idx]
best_profit = expected_profit[best_idx]

print("Optimal bid:", best_bid)
print("Expected profit at optimal:", best_profit)

# -------------------------
# Plot
# -------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=bids,
        y=expected_profit,
        mode="lines",
        name="Expected Profit"
    )
)

# optimal line
fig.add_vline(
    x=best_bid,
    line_dash="dash",
    annotation_text=f"Optimal ≈ {int(best_bid)}"
)

# your bid
fig.add_vline(
    x=2800,
    line_dash="dot",
    annotation_text="Bid=2800"
)

fig.update_layout(
    title="Expected Profit vs Bid",
    xaxis_title="Bid",
    yaxis_title="Expected Profit",
    template="plotly_white",
    width=900,
    height=500
)

fig.show()
```

    Optimal bid: 3586.466165413534
    Expected profit at optimal: 3373.590629748157
    



---
## Section 2: Expected Profit Curve

Compute E[Profit(b)] = CDF(b) × (V - b) across all bid levels.  
Find the optimal bid that maximizes expected profit.



```python
bid = 2828

p_enter = beta.cdf(
    bid / scale_max,
    alpha,
    beta_param
)

print("Entry probability at 2800:", p_enter)
```

    Entry probability at 2800: 0.610023995918379
    


```python
percentile_2800
```




    0.6035109233950711




```python
best_idx = np.argmax(expected_profit)

print("Optimal bid:", bids[best_idx])
print("Entry prob at optimal:",
      beta.cdf(
          bids[best_idx] / scale_max,
          alpha,
          beta_param
      ))
print("Max expected profit:",
      expected_profit[best_idx])
```

    Optimal bid: 3586.466165413534
    Entry prob at optimal: 0.7643740268424275
    Max expected profit: 3373.590629748157
    


```python
import numpy as np
from scipy.stats import beta

# -------------------------
# Parameters
# -------------------------

alpha = 2
beta_param = 5

scale_max = 8000
profit_gain = 8000

# bid grid
bids = np.linspace(1000, 5000, 400)

expected_profit = []

for b in bids:

    p_enter = beta.cdf(
        b / scale_max,
        alpha,
        beta_param
    )

    exp_p = p_enter * (profit_gain - b)

    expected_profit.append(exp_p)

expected_profit = np.array(expected_profit)

# optimal
best_idx = np.argmax(expected_profit)

optimal_bid = bids[best_idx]
optimal_profit = expected_profit[best_idx]

# 2800 case
bid_test = 2800

p_2800 = beta.cdf(
    bid_test / scale_max,
    alpha,
    beta_param
)

profit_2800 = p_2800 * (profit_gain - bid_test)

# regret
loss_absolute = optimal_profit - profit_2800

loss_ratio = loss_absolute / optimal_profit

print("Optimal bid:", optimal_bid)
print("Optimal expected profit:", optimal_profit)

print("2800 expected profit:", profit_2800)

print("Absolute loss:", loss_absolute)
print("Loss ratio:", loss_ratio)
```

    Optimal bid: 3395.989974937343
    Optimal expected profit: 3696.4628603421706
    2800 expected profit: 3540.78440625
    Absolute loss: 155.6784540921708
    Loss ratio: 0.0421155196126494
    

---
## Section 3: Marginal Cost Analysis

At each bid level, compute the cost per 1% additional win probability.  
The inflection point — where marginal cost accelerates — marks the efficient frontier.  
This validates that the ~3,580 optimal bid sits just below where overbidding becomes inefficient.



```python
import plotly.graph_objects as go

normalized_profit = expected_profit / optimal_profit

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=bids,
        y=normalized_profit,
        mode="lines",
        name="Profit Ratio"
    )
)

fig.add_vline(
    x=optimal_bid,
    line_dash="dash",
    annotation_text="Optimal"
)

fig.add_vline(
    x=2800,
    line_dash="dot",
    annotation_text="2800"
)

fig.update_layout(
    title="Profit Efficiency vs Bid",
    xaxis_title="Bid",
    yaxis_title="Profit / Optimal Profit",
    template="plotly_white"
)

fig.show()
```




```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

# -------------------------
# Parameters
# -------------------------

alpha = 2
beta_param = 5

scale_max = 8000

bids = np.linspace(1000, 5000, 400)

# entry probability
probs = beta.cdf(
    bids / scale_max,
    alpha,
    beta_param
)

# -------------------------
# price per probability gain
# -------------------------

delta_b = np.diff(bids)
delta_p = np.diff(probs)

price_per_prob = delta_b / delta_p

mid_probs = (probs[:-1] + probs[1:]) / 2

# -------------------------
# Plot 1: Probability vs Bid
# -------------------------

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=bids,
        y=probs,
        mode="lines",
        name="Entry Probability"
    )
)

fig1.add_vline(
    x=2800,
    line_dash="dot",
    annotation_text="Bid=2800"
)

fig1.update_layout(
    title="Entry Probability vs Bid",
    xaxis_title="Bid",
    yaxis_title="Entry Probability",
    template="plotly_white"
)

fig1.show()

# -------------------------
# Plot 2: Cost per 1% Probability
# -------------------------

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=mid_probs,
        y=price_per_prob * 0.01,
        mode="lines",
        name="Cost per 1% probability"
    )
)

fig2.update_layout(
    title="Cost to Buy Additional 1% Entry Probability",
    xaxis_title="Entry Probability",
    yaxis_title="Extra Cost per +1% Probability",
    template="plotly_white"
)

fig2.show()
```






```python
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

# -------------------------
# Parameters
# -------------------------
alpha = 2
beta_param = 5
scale_max = 8000
base_bid = 2800

# bid grid (2800 and above only)
bids = np.linspace(base_bid, 5000, 300)

# entry probabilities (via CDF)
probs = beta.cdf(bids / scale_max, alpha, beta_param)

# baseline probability
p_base = beta.cdf(base_bid / scale_max, alpha, beta_param)

# -------------------------
# Incremental cost per probability
# -------------------------
delta_bid = bids - base_bid
delta_prob = probs - p_base

# avoid division by zero
delta_prob[delta_prob <= 0] = np.nan
cost_per_prob = delta_bid / delta_prob

# cost per 1% additional win probability
cost_per_1pct = cost_per_prob * 0.01

# -------------------------
# Plot
# -------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=bids,
        y=cost_per_1pct,
        mode="lines",
        line=dict(color='firebrick', width=3),
        name="Cost per +1% Entry Prob",
        hovertemplate="<b>Bid</b>: %{x}<br><b>Cost for +1% Prob</b>: %{y:.2f}<extra></extra>"
    )
)

# Show base line
fig.add_vline(
    x=base_bid,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Base={base_bid}",
    annotation_position="top right"
)

# Add layout and grid settings
fig.update_layout(
    title={
        'text': "Extra Cost per +1% Entry Probability (Base=2800)",
        'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
    },
    xaxis=dict(
        title="Bid Amount",
        showgrid=True,
        gridcolor='lightgray', # major grid color
        gridwidth=1,
        zeroline=False
    ),
    yaxis=dict(
        title="Extra Cost per +1% Probability",
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        minor=dict(showgrid=True, gridcolor='#f0f0f0') # add minor grid
    ),
    template="plotly_white",
    width=900,
    height=550,
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()
```




```python
# set threshold
threshold = 0.05

# find first bid satisfying condition
mask = distance >= threshold

if np.any(mask):

    first_idx = np.argmax(mask)

    bid_threshold = bids[first_idx]
    distance_threshold = distance[first_idx]

    print("First bid where distance >= 0.05:", bid_threshold)
    print("Distance there:", distance_threshold)

else:

    bid_threshold = None
    print("No point exceeds threshold")

# -------------------------
# Plot
# -------------------------

fig2 = go.Figure()

# main curve
fig2.add_trace(
    go.Scatter(
        x=bids,
        y=distance,
        mode="lines",
        name="Distance from Linear Growth"
    )
)

# horizontal threshold line
fig2.add_hline(
    y=threshold,
    line_dash="dash",
    annotation_text="distance = 0.05"
)

# marker: first crossing point
if bid_threshold is not None:

    fig2.add_trace(
        go.Scatter(
            x=[bid_threshold],
            y=[distance_threshold],
            mode="markers",
            marker=dict(size=10),
            name="First crossing"
        )
    )

fig2.update_layout(
    title="Distance from Linear with Threshold Marker",
    xaxis_title="Bid",
    yaxis_title="Extra Probability vs Linear",
    template="plotly_white"
)

fig2.show()
```

    First bid where distance >= 0.05: 3579.9331103678933
    Distance there: 0.05012018506480409
    




```python
fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=bids,
        y=probs - p_base,
        mode="lines",
        name="Extra Entry Probability"
    )
)

fig2.update_layout(
    title="Extra Entry Probability vs Bid (Base=2800)",
    xaxis_title="Bid",
    yaxis_title="Extra Probability",
    template="plotly_white"
)

fig2.show()
```




```python

```
