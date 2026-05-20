# Round 2 Manual: Three-Pillar Resource Allocation Optimization

> **Budget:** 50,000 XIRECS  
> **Pillars:** Research (logarithmic) · Scale (linear) · Speed (rank-based)  
> **Core problem:** Speed payoff depends on the field's allocation distribution — modeled as Beta(α, β).

## Overview

The manual challenge required allocating a fixed budget across three pillars with different marginal return structures.  
Speed was rank-based: its payoff depended entirely on competitors' allocations, making distributional modeling critical.

**Approach:**
1. Fix Speed allocation z, optimize Research/Scale split for the remaining budget
2. Model Speed distribution as Beta(α, β); compute expected Speed multiplier via numerical integration
3. Penalize sensitivity to (α, β) misspecification in the objective — risk-adjusted optimization

**Final allocation:** Research 16% (8,000) · Scale 50% (25,000) · Speed 34% (17,000)  
**Manual P&L:** 183,999 XIRECS


---
## Section 1: A×B Optimization (Research × Scale)

For a fixed Speed allocation z, find the Research/Scale split that maximizes:
```
Research(x_R) × Scale(x_S)
```
subject to x_R + x_S = 50,000 - z.

Research follows a logarithmic growth curve; Scale is linear. Optimal split is derived analytically.



```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

##################################
# A×B optimization (Research × Scale)
##################################

def get_max_ab(z):

    remaining = 100 - z

    if remaining <= 0:
        return 0, 0, 0

    def objective(x):

        y = remaining - x

        A = 200000 * np.log(1 + x) / np.log(101)
        B = 0.07 * y

        return -(A * B)

    res = minimize_scalar(
        objective,
        bounds=(0, remaining),
        method='bounded'
    )

    opt_x = res.x
    opt_y = remaining - opt_x

    return -res.fun, opt_x, opt_y


##################################
# Full PnL
##################################

def get_full_pnl(z, alpha, beta_param):

    ab_val, x, y = get_max_ab(z)

    rank_p = beta.cdf(
        z / 100,
        alpha,
        beta_param
    )

    C_score = 0.1 + 0.8 * rank_p

    pnl = ab_val * C_score

    return pnl


##################################
# Robust Objective
##################################

def robust_objective(z, alpha, beta_param):

    z = float(z)

    base = get_full_pnl(
        z,
        alpha,
        beta_param
    )

    # alpha ± 0.1
    pnl_a_plus = get_full_pnl(
        z,
        alpha + 0.2,
        beta_param
    )

    pnl_a_minus = get_full_pnl(
        z,
        alpha - 0.2,
        beta_param
    )

    avg_a = (
        pnl_a_plus +
        pnl_a_minus
    ) / 2

    alpha_risk = abs(
        base - avg_a
    )

    # beta ± 0.1
    pnl_b_plus = get_full_pnl(
        z,
        alpha,
        beta_param + 0.2
    )

    pnl_b_minus = get_full_pnl(
        z,
        alpha,
        beta_param - 0.2
    )

    avg_b = (
        pnl_b_plus +
        pnl_b_minus
    ) / 2

    beta_risk = abs(
        base - avg_b
    )

    denom = (
        alpha_risk *
        beta_risk +
        1e-9
    )

    score = base / denom

    return -score


##################################
# Run
##################################

alpha = 1.75
beta_param = 5.0

res = minimize_scalar(
    robust_objective,
    bounds=(0, 60),
    args=(alpha, beta_param),
    method='bounded'
)

opt_z = res.x

##################################
# Compute result
##################################

ab_val, opt_x, opt_y = get_max_ab(opt_z)

rank_p = beta.cdf(
    opt_z / 100,
    alpha,
    beta_param
)

C_score = 0.1 + 0.8 * rank_p

final_pnl = ab_val * C_score

##################################
# Print output
##################################

print("----- ROBUST RESULT -----")

print(f"A ≈ {opt_x:.2f}%")
print(f"B ≈ {opt_y:.2f}%")
print(f"C ≈ {opt_z:.2f}%")

print(f"C Score ≈ {C_score:.4f}")

print(f"Total PnL ≈ {final_pnl - 50000:.2f}")
```

    ----- ROBUST RESULT -----
    A ≈ 16.97%
    B ≈ 51.91%
    C ≈ 31.12%
    C Score ≈ 0.6326
    Total PnL ≈ 237738.24
    


```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import beta

##################################
# 1. Inner optimization: maximize A×B given fixed z
##################################
def get_max_ab(z):
    remaining = 100 - z
    if remaining <= 0: return 0, 0, 0
    
    # Find x split to maximize A×B (y = remaining − x)
    def objective_ab(x):
        y = remaining - x
        A = 200000 * np.log(1 + x) / np.log(101)
        B = 0.07 * y
        return -(A * B)
    
    # 1D optimization over x
    res_inner = minimize_scalar(objective_ab, bounds=(0.1, remaining-0.1), method='bounded')
    opt_x = res_inner.x
    opt_y = remaining - opt_x
    return -res_inner.fun, opt_x, opt_y

def get_pnl_full(z, alpha, b_param):
    ab_max, x, y = get_max_ab(z)
    rank_p = beta.cdf(z / 100, alpha, b_param)
    C_expected = 0.1 + 0.8 * rank_p
    return ab_max * C_expected, ab_max, rank_p, x, y

##################################
# 2. Ranking Sharpe objective (z as sole variable)
##################################
def ranking_sharpe_objective(z, alpha, b_param):
    z = float(z)
    curr_pnl, ab_max, _, _, _ = get_pnl_full(z, alpha, b_param)
    
    # Compute change in optimal A×B per 1% change in z (opportunity cost)
    delta = 1.0
    ab_up, _, _ = get_max_ab(z + delta if z + delta <= 100 else z)
    ab_down, _, _ = get_max_ab(z - delta if z - delta >= 0 else z)
    
    rank_risk = (abs(ab_max - ab_up) + abs(ab_max - ab_down)) / 2
    return -(curr_pnl / (rank_risk + 1e-9))

##################################
# 3. Run optimization and extract result
##################################
base_alpha, base_beta = 1.75, 5.0
res = minimize_scalar(ranking_sharpe_objective, bounds=(1.0, 99.0), args=(base_alpha, base_beta), method='bounded')

opt_z = res.x
final_pnl, max_ab, final_threshold, opt_x, opt_y = get_pnl_full(opt_z, base_alpha, base_beta)

print(f"--- Refined optimization result ---")
print(f"A: {opt_x:.2f}%, B: {opt_y:.2f}%, C: {opt_z:.2f}%")
print(f"P&L: {final_pnl - 50000:.2f} | Threshold: top {(1-final_threshold)*100:.2f}%")

##################################
# 4. Sensitivity visualization
##################################
a_grid = np.linspace(1.0, 4.0, 50)
b_grid = np.linspace(2.0, 8.0, 50)

# Track P&L change as Beta parameters vary (optimal z fixed)
pnl_vs_a = [get_pnl_full(opt_z, a, base_beta)[0] for a in a_grid]
pnl_vs_b = [get_pnl_full(opt_z, base_alpha, b)[0] for b in b_grid]
sens_a = np.gradient(pnl_vs_a, a_grid)
sens_b = np.gradient(pnl_vs_b, b_grid)

fig = make_subplots(rows=2, cols=2, subplot_titles=("PnL vs Alpha", "Alpha Sensitivity (Delta)", "PnL vs Beta", "Beta Sensitivity (Delta)"))
fig.add_trace(go.Scatter(x=a_grid, y=pnl_vs_a, name="PnL(a)", line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=a_grid, y=sens_a, name="dPnL/da", line=dict(color='red', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=b_grid, y=pnl_vs_b, name="PnL(b)", line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=b_grid, y=sens_b, name="dPnL/db", line=dict(color='orange', dash='dash')), row=2, col=2)

fig.update_layout(height=800, title=f"<b>Final Strategy Analysis (Fixed C={opt_z:.1f}%)</b>", template="plotly_white", showlegend=False)
fig.show()
```

    --- 정교화된 최적화 결과 ---
    A: 15.69%, B: 46.98%, C: 37.32%
    PnL: 238032.78 | Threshold: 상위 22.76%
    



---
## Section 2: Beta Distribution Modeling for Speed

Speed multiplier is rank-based: modeled by fitting a Beta(α, β) distribution to competitors' Speed allocations.

Parameters: α=1.5, β=5 — reflecting moderate concentration in the low-to-mid range with a long right tail.

**Sensitivity analysis:** P&L was materially more sensitive to β than α. Higher Speed allocation reduced this sensitivity — the risk-adjusted objective pushed toward elevated Speed investment.



```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
import plotly.graph_objects as go

##################################
# Optimization (Beta Distribution)
##################################

def optimize_allocation(a, b_param):

    def objective(p):
        x, y, z = p
        # Soft constraint: sum ≤ 100 (SLSQP handles this well)
        if x + y + z > 100.01: return 1e10

        A = 200000 * np.log(1 + x) / np.log(101)
        B = 0.07 * y

        # Convert z allocation (0–100) to fraction (0–1)
        z_ratio = np.clip(z / 100, 1e-9, 1-1e-9)

        # Estimate rank using Beta CDF
        rank_pct = beta.cdf(z_ratio, a, b_param)

        C = 0.1 + 0.8 * rank_pct
        pnl = A * B * C
        return -pnl

    # Constraint: x + y + z ≤ 100
    cons = ({'type': 'ineq', 'fun': lambda p: 100 - np.sum(p)})
    bounds = [(0, 100), (0, 100), (0, 100)]

    res = minimize(
        objective,
        [33.3, 33.3, 33.4], # Initial value: uniform allocation
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
    return res.x

##################################
# PnL evaluator
##################################

def compute_pnl(p, a, b_param):
    x, y, z = p
    A = 200000 * np.log(1 + x) / np.log(101)
    B = 0.07 * y
    z_ratio = np.clip(z / 100, 1e-9, 1-1e-9)
    
    rank_pct = beta.cdf(z_ratio, a, b_param)
    C = 0.1 + 0.8 * rank_pct
    
    pnl = A * B * C - 50000
    return pnl

##################################
# Assumed model parameters (α, β)
##################################

a_assumed = 1.75
b_assumed = 5.0

x_assumed = optimize_allocation(a_assumed, b_assumed)

##################################
# True parameter grid (actual α, β)
##################################

# Search α in [1.0, 3.0] (aggression), β in [3.0, 7.0] (conservatism)
a_grid = np.linspace(1.0, 3.0, 25)
b_grid = np.linspace(3.0, 7.0, 25)

regret = np.zeros((len(b_grid), len(a_grid)))

for i, b_t in enumerate(b_grid):
    for j, a_t in enumerate(a_grid):

        # 1. True optimal solution under actual parameters (a_t, b_t)
        x_true_opt = optimize_allocation(a_t, b_t)
        pnl_true_max = compute_pnl(x_true_opt, a_t, b_t)

        # 2. Apply strategy from assumed parameters to actual parameter environment
        pnl_from_assumed = compute_pnl(x_assumed, a_t, b_t)

        # Regret = max achievable P&L − strategy P&L under assumed params
        regret[i, j] = pnl_true_max - pnl_from_assumed

##################################
# Plot regret heatmap
##################################

fig = go.Figure(
    data=go.Heatmap(
        z=regret,
        x=a_grid,
        y=b_grid,
        colorscale="Reds",
        colorbar=dict(title="Regret (Opportunity Loss)")
    )
)

fig.update_layout(
    title=f"Regret Map: Assumed (α={a_assumed}, β={b_assumed}) vs True Market",
    xaxis_title="True alpha (Market Aggressiveness)",
    yaxis_title="True beta (Market Conservativeness)",
    template="plotly_white"
)

fig.show()
```



---
## Section 3: Grid Search & Final Allocation

Grid search over (α, β, z) with risk-adjusted target function:
```
Target(α, β) = P&L*(α, β) / (α_risk × β_risk)
```
where α_risk and β_risk measure sensitivity to 0.1-unit perturbations in the Beta parameters.



```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta
import plotly.graph_objects as go

##################################
# A×B optimization (Research × Scale)
##################################

def get_max_ab(z):

    remaining = 100 - z

    if remaining <= 0:
        return 0, 0, 0

    def objective(x):

        y = remaining - x

        A = 200000 * np.log(1 + x) / np.log(101)
        B = 0.07 * y

        return -(A * B)

    res = minimize_scalar(
        objective,
        bounds=(0, remaining),
        method='bounded'
    )

    opt_x = res.x
    opt_y = remaining - opt_x

    return -res.fun, opt_x, opt_y


##################################
# Full PnL
##################################

def get_full_pnl(z, alpha, beta_param):

    ab_val, x, y = get_max_ab(z)

    rank_p = beta.cdf(
        z / 100,
        alpha,
        beta_param
    )

    C_score = 0.1 + 0.8 * rank_p

    pnl = ab_val * C_score

    return pnl


##################################
# Robust Objective
##################################

def robust_objective(z, alpha, beta_param):

    z = float(z)

    base = get_full_pnl(
        z,
        alpha,
        beta_param
    )

    delta = 0.2

    # alpha risk
    pnl_a_plus = get_full_pnl(
        z,
        alpha + delta,
        beta_param
    )

    pnl_a_minus = get_full_pnl(
        z,
        alpha - delta,
        beta_param
    )

    avg_a = (pnl_a_plus + pnl_a_minus) / 2

    alpha_risk = abs(
        base - avg_a
    )

    # beta risk
    pnl_b_plus = get_full_pnl(
        z,
        alpha,
        beta_param + delta
    )

    pnl_b_minus = get_full_pnl(
        z,
        alpha,
        beta_param - delta
    )

    avg_b = (pnl_b_plus + pnl_b_minus) / 2

    beta_risk = abs(
        base - avg_b
    )

    denom = (
        alpha_risk *
        beta_risk +
        1e-9
    )

    score = base / denom

    return -score


##################################
# Run
##################################

alpha = 2
beta_param = 5.0

res = minimize_scalar(
    robust_objective,
    bounds=(0, 60),
    args=(alpha, beta_param),
    method='bounded'
)

opt_z = res.x


##################################
# Compute result
##################################

ab_val, opt_x, opt_y = get_max_ab(opt_z)

rank_p = beta.cdf(
    opt_z / 100,
    alpha,
    beta_param
)

C_score = 0.1 + 0.8 * rank_p

final_pnl = ab_val * C_score

threshold_pct = (1 - rank_p) * 100


##################################
# Print output
##################################

print("----- ROBUST RESULT -----")

print(f"A ≈ {opt_x:.2f}%")
print(f"B ≈ {opt_y:.2f}%")
print(f"C ≈ {opt_z:.2f}%")

print(f"C Score ≈ {C_score:.4f}")

print(f"Expected PnL ≈ {final_pnl - 50000:,.0f}")

print(f"Target Threshold ≈ top {threshold_pct:.2f}%")


##################################
# Ranking Sharpe (display)
##################################

base = final_pnl

delta = 0.2

pnl_a_plus = get_full_pnl(opt_z, alpha+delta, beta_param)
pnl_a_minus = get_full_pnl(opt_z, alpha-delta, beta_param)

pnl_b_plus = get_full_pnl(opt_z, alpha, beta_param+delta)
pnl_b_minus = get_full_pnl(opt_z, alpha, beta_param-delta)

risk = np.std([
    pnl_a_plus,
    pnl_a_minus,
    pnl_b_plus,
    pnl_b_minus
])

ranking_sharpe = base / (risk + 1e-9)


##################################
# Plotly Donut Chart
##################################

fig = go.Figure()

fig.add_trace(
    go.Pie(
        labels=[
            "A (Log Asset)",
            "B (Linear Asset)",
            "C (Rank Asset)"
        ],

        values=[
            opt_x,
            opt_y,
            opt_z
        ],

        hole=0.55,

        textinfo="label+percent",

        marker=dict(
            colors=[
                "#636EFA",  # A
                "#EF553B",  # B
                "#00CC96"   # C
            ]
        )
    )
)

##################################
# Center text
##################################

fig.add_annotation(
    text="Asset Allocation",
    x=0.5,
    y=0.5,
    showarrow=False,
    font=dict(size=16)
)

##################################
# Bottom metric text
##################################

metric_text = (
    f"<b>Expected P&L:</b> {final_pnl - 50000:,.0f}<br>"
    f"<b>Target Threshold:</b> top {threshold_pct:.2f}%<br>"
    f"<b>Ranking Sharpe:</b> {ranking_sharpe:.4f}"
)

fig.add_annotation(
    text=metric_text,
    x=0.5,
    y=-0.15,
    showarrow=False,
    align="center",
    font=dict(size=15)
)

##################################
# Layout
##################################

fig.update_layout(

    title=dict(
        text="<b>Final Robust Strategy Dashboard (Nested Optimization)</b>",
        x=0.5
    ),

    showlegend=True,

    margin=dict(
        t=80,
        b=100
    )
)

fig.show()
```

    ----- ROBUST RESULT -----
    A ≈ 16.40%
    B ≈ 49.70%
    C ≈ 33.90%
    C Score ≈ 0.6280
    Expected PnL ≈ 220,432
    Target Threshold ≈ 상위 34.00%
    




```python

```
