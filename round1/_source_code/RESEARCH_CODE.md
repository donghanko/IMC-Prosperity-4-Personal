# Round 1: Alpha Generation & Market Making

> **Assets:** INTARIAN_PEPPER_ROOT · ASH_COATED_OSMIUM  
> **Theme:** Two delta-1 assets, pure market making. Core task: fair value estimation.

## Overview

Round 1 was the foundation round — no options, no exotic structures. The core challenge was fair value construction: without a reliable anchor, any quoting strategy is noise.

Key findings:
- **PEPPER** required a dynamic, trend-adjusted fair value anchor (linear regression on price drift).
- **OSMIUM** required a stable mean-reversion anchor; instantaneous mid tracked regime shifts too closely.
- **Spread-break arbitrage** appeared structurally riskless but failed due to toxic counterparties injected by the competition design.


---
## Section 1: INTARIAN_PEPPER_ROOT — Initial Exploration

Cells 00–06: Load price and trade data across all three days. Visualize mid-price distribution and spread dynamics per day.

> **Finding:** PEPPER exhibits a clear persistent linear upward price trend across all three days. Static fair value anchors (rolling mid, VWAP) would be systematically stale.



```python
import pandas as pd
import glob
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Data loading setup
price_files = sorted(glob.glob('prices_round_1_day_*.csv'))
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

# Load price data (includes mid_price)
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# Load trade data (includes trade price)
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. Visualize per asset
assets = prices['product'].unique()

for asset in assets:
    # Filter data for target asset
    asset_prices = prices[prices['product'] == asset][['global_timestamp', 'mid_price']].copy()
    asset_trades = trades[trades['symbol'] == asset][['global_timestamp', 'price']].copy()
    asset_trades.rename(columns={'price': 'trade_price'}, inplace=True)
    
    # Merge price and trade data (by timestamp)
    # Match trade prices to all price timestamps
    combined = pd.merge(asset_prices, asset_trades, on='global_timestamp', how='left')
    
    # Handle duplicate timestamps (last trade wins if multiple trades at same timestamp)
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    # Forward-fill missing values
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill() # forward-fill mid_price if also missing
    
    # 3. Create interactive Plotly chart
    fig = go.Figure()

    # Mid-price line
    fig.add_trace(go.Scatter(
        x=combined['global_timestamp'],
        y=combined['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='rgba(100, 149, 237, 0.8)', width=1.5)
    ))

    # Trade price line (dot/dashed style)
    fig.add_trace(go.Scatter(
        x=combined['global_timestamp'],
        y=combined['trade_price'],
        mode='lines',
        name='Trade Price (Interpolated)',
        line=dict(color='rgba(255, 99, 71, 0.7)', width=1, dash='dot')
    ))

    fig.update_layout(
        title=f'Price Analysis: {asset} (Mid vs Trade)',
        xaxis_title='Global Timestamp (Day * 1M + T)',
        yaxis_title='Price',
        template='plotly_dark', # Dark mode for premium look
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()

```






```python
import pandas as pd
import glob
import os
import re
import numpy as np
import plotly.graph_objects as go

# 1. Data loading setup
price_files = sorted(glob.glob('prices_round_1_day_*.csv'))
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

# Load price data
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# Load trade data
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. Process and visualize per asset
assets = prices['product'].unique()

for asset in assets:
    asset_prices = prices[prices['product'] == asset][['global_timestamp', 'mid_price']].copy()
    asset_trades = trades[trades['symbol'] == asset][['global_timestamp', 'price']].copy()
    asset_trades.rename(columns={'price': 'trade_price'}, inplace=True)
    
    # Replace zero values with NaN for interpolation
    asset_prices['mid_price'] = asset_prices['mid_price'].replace(0, np.nan)
    asset_trades['trade_price'] = asset_trades['trade_price'].replace(0, np.nan)
    
    # Merge data
    combined = pd.merge(asset_prices, asset_trades, on='global_timestamp', how='left')
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    # Forward-fill interpolation
    # Zeros and unmatched timestamps are filled with the last valid value.
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill()
    
    # Back-fill any leading NaNs (optional)
    combined['trade_price'] = combined['trade_price'].bfill()
    combined['mid_price'] = combined['mid_price'].bfill()
    
    # 3. Visualization
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined['global_timestamp'],
        y=combined['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='rgba(100, 149, 237, 0.9)', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=combined['global_timestamp'],
        y=combined['trade_price'],
        mode='lines',
        name='Trade Price (Cleaned)',
        line=dict(color='rgba(255, 99, 71, 0.8)', width=1, dash='dot')
    ))

    fig.update_layout(
        title=f'Price Analysis: {asset} (Zero values Interpolated)',
        xaxis_title='Global Timestamp',
        yaxis_title='Price',
        template='plotly_dark',
        hovermode='x unified'
    )

    fig.show()

```






```python
import pandas as pd
import glob
import re
import numpy as np
import plotly.graph_objects as go

# 1. Data loading (same as above)
price_files = sorted(glob.glob('prices_round_1_day_*.csv'))
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

# Load price data
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# Load trade data
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. Define fair value
def get_fair_value(asset, t):
    if 'PEPPER' in asset:
        return 10000 + 0.001 * t
    elif 'OSMIUM' in asset:
        return 10000
    else:
        return np.nan

# 3. Visualize per asset (trade deviation as scatter points)
assets = prices['product'].unique()

for asset in assets:
    # Extract asset trade data and remove zero-price entries
    asset_trades = trades[trades['symbol'] == asset].copy()
    asset_trades = asset_trades[asset_trades['price'] > 0] # Remove zero-price fills (likely spurious)
    
    # [Fair value & deviation] — use t at actual trade timestamp
    asset_trades['fair_value'] = asset_trades['global_timestamp'].apply(lambda t: get_fair_value(asset, t))
    asset_trades['deviation'] = asset_trades['price'] - asset_trades['fair_value']
    
    # Reference: mid-price deviation (faint background line)
    asset_prices = prices[prices['product'] == asset].copy()
    asset_prices = asset_prices[asset_prices['mid_price'] > 0]
    asset_prices['fair_value'] = asset_prices['global_timestamp'].apply(lambda t: get_fair_value(asset, t))
    asset_prices['deviation'] = asset_prices['mid_price'] - asset_prices['fair_value']

    # 4. Visualization
    fig = go.Figure()

    # Mid-price deviation (faint background line)
    fig.add_trace(go.Scatter(
        x=asset_prices['global_timestamp'],
        y=asset_prices['deviation'],
        mode='lines',
        name='Mid Price Deviation',
        line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
    ))

    # Trade deviation (scatter points)
    fig.add_trace(go.Scatter(
        x=asset_trades['global_timestamp'],
        y=asset_trades['deviation'],
        mode='markers',
        name='Actual Trades',
        marker=dict(
            size=4,
            color='rgba(0, 255, 127, 0.7)',
            symbol='circle'
        )
    ))

    # Zero line (fair value reference)
    fig.add_shape(
        type="line", line=dict(color="red", width=1, dash="dash"),
        x0=asset_prices['global_timestamp'].min(), x1=asset_prices['global_timestamp'].max(),
        y0=0, y1=0
    )

    fig.update_layout(
        title=f'Trade Deviation Points: {asset} (Actual Trade - Fair)',
        xaxis_title='Global Timestamp',
        yaxis_title='Price Deviation',
        template='plotly_dark',
        hovermode='closest'
    )

    fig.show()

```

    c:\Users\dhko23\AppData\Local\anaconda3\Lib\site-packages\pandas\core\arrays\masked.py:61: UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).
      from pandas.core import (
    






```python
import pandas as pd
import glob
import plotly.graph_objects as go

# 1. Load data
price_files = sorted(glob.glob('prices_round_1_day_*.csv'))

all_prices = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    all_prices.append(df)
prices = pd.concat(all_prices)

# Compute continuous timestamp
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# 2. Filter OSMIUM (ASH_COATED_OSMIUM) data
product = "ASH_COATED_OSMIUM"
osmium_data = prices[prices['product'] == product].copy()

# Handle zeros (missing data): replace with NaN and forward-fill
for col in ['bid_price_1', 'ask_price_1', 'mid_price']:
    osmium_data[col] = osmium_data[col].replace(0, pd.NA).ffill().bfill()

# 3. Visualization
fig = go.Figure()

# Ask Price L1 (offer side — typically above mid)
fig.add_trace(go.Scatter(
    x=osmium_data['global_timestamp'],
    y=osmium_data['ask_price_1'],
    mode='lines',
    name='Ask Price 1',
    line=dict(color='rgba(255, 99, 71, 0.8)', width=1)
))

# Mid price
fig.add_trace(go.Scatter(
    x=osmium_data['global_timestamp'],
    y=osmium_data['mid_price'],
    mode='lines',
    name='Mid Price',
    line=dict(color='rgba(255, 255, 255, 0.9)', width=1.5)
))

# Bid Price L1 (bid side — typically below mid)
fig.add_trace(go.Scatter(
    x=osmium_data['global_timestamp'],
    y=osmium_data['bid_price_1'],
    mode='lines',
    name='Bid Price 1',
    line=dict(color='rgba(100, 149, 237, 0.8)', width=1)
))

fig.update_layout(
    title=f'Market Depth Analysis: {product} (Bid, Ask, Mid)',
    xaxis_title='Global Timestamp',
    yaxis_title='Price',
    template='plotly_dark',
    hovermode='x unified',
    # Add slider for zooming into specific intervals
    xaxis=dict(rangeslider=dict(visible=True)) 
)

fig.show()

```

    /var/folders/4r/hpt0pryx6tq5141_h_9c3r740000gn/T/ipykernel_91291/3261667570.py:23: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      osmium_data[col] = osmium_data[col].replace(0, pd.NA).ffill().bfill()
    




```python
import pandas as pd
import glob
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# 1. Load data
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

all_trades = []
for f in trade_files:
    df = pd.read_csv(f, sep=';')
    all_trades.append(df)
trades = pd.concat(all_trades)

# 2. Filter OSMIUM data and compute price deviation
product = "ASH_COATED_OSMIUM"
base_mid = 10000
osmium_trades = trades[trades['symbol'] == product].copy()
osmium_trades = osmium_trades[osmium_trades['price'] > 0] # valid prices only

# Compute deviation from reference price (10,000)
osmium_trades['deviation'] = osmium_trades['price'] - base_mid

# 3. PDF (Probability Density Function) visualization
# Plot histogram with KDE overlay.
hist_data = [osmium_trades['deviation'].tolist()]
group_labels = ['Trade Deviation (Price - 10000)']

fig = ff.create_distplot(
    hist_data, 
    group_labels, 
    bin_size=1, # OSMIUM prices are integer-valued — bin size = 1
    show_hist=True,
    show_rug=False,
    colors=['#00FF7F']
)

fig.update_layout(
    title=f'PDF of Taker Trades for {product} (Basis: {base_mid})',
    xaxis_title='Price Deviation (Trade Price - 10000)',
    yaxis_title='Density',
    template='plotly_dark',
    bargap=0.05
)

# Show zero reference line
fig.add_shape(
    type="line", line=dict(color="red", width=2, dash="dash"),
    x0=0, x1=0, y0=0, y1=1,
    yref='paper'
)

fig.show()

```




```python
import pandas as pd
import glob
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# 1. Load data
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

all_trades = []
for f in trade_files:
    df = pd.read_csv(f, sep=';')
    all_trades.append(df)
trades = pd.concat(all_trades)

# 2. Filter OSMIUM data and classify buy/sell
product = "ASH_COATED_OSMIUM"
base_mid = 10000
osmium_trades = trades[trades['symbol'] == product].copy()
osmium_trades = osmium_trades[osmium_trades['price'] > 0]

# Compute price deviation
osmium_trades['deviation'] = osmium_trades['price'] - base_mid

# Buy: fill price > 10,000 (typically executed at ask 10002–10005)
# Sell: fill price < 10,000 (typically executed at bid 9995–9998)
buy_trades = osmium_trades[osmium_trades['deviation'] > 0]['deviation'].tolist()
sell_trades = osmium_trades[osmium_trades['deviation'] < 0]['deviation'].tolist()

# 3. PDF visualization (Buy: red, Sell: blue)
hist_data = [buy_trades, sell_trades]
group_labels = ['Taker Buy (Price > 10000)', 'Taker Sell (Price < 10000)']
colors = ['#FF4136', '#0074D9'] # Red for Buy, Blue for Sell

# Error handling for empty data
if len(buy_trades) > 0 and len(sell_trades) > 0:
    fig = ff.create_distplot(
        hist_data, 
        group_labels, 
        bin_size=1, 
        show_hist=True,
        show_rug=False,
        colors=colors
    )

    fig.update_layout(
        title=f'PDF of Osmium Trades: Buy vs Sell (Basis: {base_mid})',
        xaxis_title='Price Deviation (Trade Price - 10000)',
        yaxis_title='Density',
        template='plotly_dark',
        bargap=0.1
    )

    # Show 10,000 reference line
    fig.add_shape(
        type="line", line=dict(color="white", width=2, dash="dash"),
        x0=0, x1=0, y0=0, y1=1,
        yref='paper'
    )

    fig.show()
else:
    print("Insufficient buy/sell trade data to analyze.")

```




```python
import pandas as pd
import glob
import numpy as np
import plotly.figure_factory as ff
import re

# 1. Load and preprocess data
trade_files = sorted(glob.glob('trades_round_1_day_*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

all_trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    all_trades_list.append(df)
trades = pd.concat(all_trades_list)

# 2. Filter PEPPER data and build continuous timestamp
product = "INTARIAN_PEPPER_ROOT"
# Compute continuous t (global timestamp) relative to start of dataset (typically day -2)
day_min = trades['day'].min() 
trades['global_ts'] = (trades['day'] - day_min) * 1000000 + trades['timestamp']

pepper_trades = trades[trades['symbol'] == product].copy()
pepper_trades = pepper_trades[pepper_trades['price'] > 0]

# Compute fair value deviation: Fair = 10,000 + 0.001·t
pepper_trades['fair_value'] = 10000 + 0.001 * pepper_trades['global_ts']
pepper_trades['deviation'] = pepper_trades['price'] - pepper_trades['fair_value']

# Buy (red): fill > Fair; Sell (blue): fill < Fair
buy_trades = pepper_trades[pepper_trades['deviation'] > 0]['deviation'].tolist()
sell_trades = pepper_trades[pepper_trades['deviation'] < 0]['deviation'].tolist()

# 3. PDF Visualization
if len(buy_trades) > 0 and len(sell_trades) > 0:
    hist_data = [buy_trades, sell_trades]
    group_labels = ['Pepper Taker Buy (Price > Fair)', 'Pepper Taker Sell (Price < Fair)']
    colors = ['#FF4136', '#0074D9'] 

    # Fixed typo: create_distplot
    fig = ff.create_distplot(
        hist_data, 
        group_labels, 
        bin_size=1, 
        show_hist=True,
        show_rug=False,
        colors=colors
    )

    fig.update_layout(
        title=f'PDF of Pepper Trades relative to Fair Value (10000 + 0.001t)',
        xaxis_title='Price Deviation (Trade Price - Fair Value)',
        yaxis_title='Density',
        template='plotly_dark',
        bargap=0.1
    )

    # Show zero reference line
    fig.add_shape(
        type="line", line=dict(color="white", width=2, dash="dash"),
        x0=0, x1=0, y0=0, y1=1,
        yref='paper'
    )

    fig.show()
else:
    print("Insufficient buy/sell trade data to analyze.")

```



---
## Section 2: PEPPER — Trend-Following Fair Value

Cells 07–14: Detailed interval-level analysis of PEPPER mid-price. Trade-type overlays (active buy/sell, passive fills) confirm the trend direction.

**Fair value construction:**
```
FV_t = α·t + β
```
where α (slope) and β (intercept) are estimated via linear regression on historical mid-prices.
Quotes are centered on FV_t — not on the instantaneous mid — to avoid adverse selection on a drifting asset.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# 1. Load and filter PEPPER data
def load_pepper_data():
    files = sorted(glob.glob('prices_round_1_day_*.csv'))
    df = pd.concat([pd.read_csv(f, sep=';') for f in files])
    # Filter PEPPER ROOT data and sort by timestamp
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    pepper = pepper.sort_values(['day', 'timestamp']).reset_index(drop=True)
    return pepper

df = load_pepper_data()

# 2. Compute 'intent' and 'atmosphere' signals
# Micro-price: size-weighted mid price capturing order book 'intent'
df['micro_price'] = (df['bid_price_1'] * df['ask_volume_1'] + 
                     df['ask_price_1'] * df['bid_volume_1']) / \
                    (df['bid_volume_1'] + df['ask_volume_1'])

# OBI (Order Book Imbalance): (bid_size - ask_size) / (bid_size + ask_size), range [-1, 1]
df['obi'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])

# Exponential moving average for 'atmosphere' — slow-moving trend
df['obi_ema'] = df['obi'].ewm(span=100).mean()
df['intent_drift'] = (df['micro_price'] - df['mid_price']).rolling(50).mean()

# 3. Premium visualization (verify signal alignment)
plt.style.use('seaborn-v0_8-muted') # Apply clean seaborn theme
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Chart 1: Price vs. micro-price (does intent lead price?)
ax1.plot(df['timestamp'], df['mid_price'], label='Mid Price', color='#bdc3c7', alpha=0.5)
ax1.plot(df['timestamp'], df['micro_price'], label='Micro Price (Intent)', color='#2980b9', linewidth=1)
ax1.set_title('Intarian Pepper Root: Price & Hidden Intent', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.2)

# Chart 2: Market 'atmosphere' (which way is the order book tilted?)
ax2.fill_between(df['timestamp'], 0, df['obi_ema'], where=(df['obi_ema'] >= 0), color='#2ecc71', alpha=0.3, label='Buy Lean')
ax2.fill_between(df['timestamp'], 0, df['obi_ema'], where=(df['obi_ema'] < 0), color='#e74c3c', alpha=0.3, label='Sell Lean')
ax2.axhline(0, color='black', alpha=0.2, linestyle='--')
ax2.set_title('Market Atmosphere (Order Book Imbalance EMA)', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.2)

# Chart 3: Hidden signal (accumulated divergence between intent and realized price)
ax3.plot(df['timestamp'], df['intent_drift'], color='#8e44ad', label='Cumulative Signal (Micro - Mid)')
ax3.fill_between(df['timestamp'], 0, df['intent_drift'], color='#8e44ad', alpha=0.1)
ax3.axhline(0, color='black', alpha=0.2)
ax3.set_title('Hidden Signal: The Moment of Imbalance', fontsize=12)
ax3.set_xlabel('Timestamp')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# Note: if Chart 3 (purple) diverges from zero while price (Chart 1) stays flat, 
# a price move in that direction may be imminent.

```


    
![png](round1_analysis_files/round1_analysis_10_0.png)
    



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_interpolated_pepper_days():
    # 1. Load data per day (day -2, -1, 0)
    days = [-2, -1, 0]
    fig, axes = plt.subplots(len(days), 1, figsize=(15, 12), sharex=True)
    
    for i, day in enumerate(days):
        # Read file
        filename = f'prices_round_1_day_{day}.csv'
        try:
            df_day = pd.read_csv(filename, sep=';')
            pepper = df_day[df_day['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
            
            # Generate full timestamp grid (0–999,900) and interpolate
            all_timestamps = np.arange(0, 1000000, 100)
            full_index = pd.DataFrame({'timestamp': all_timestamps})
            pepper = pd.merge(full_index, pepper, on='timestamp', how='left')
            
            # Apply linear interpolation
            pepper['mid_price'] = pepper['mid_price'].interpolate(method='linear')
            
            # 2. Data visualization
            axes[i].plot(pepper['timestamp'], pepper['mid_price'], label=f'Day {day} Mid Price', color='#3498db', alpha=0.7)
            
            # 3. Trend check (add linear regression line)
            # Visually confirm PEPPER's slow upward drift property
            z = np.polyfit(pepper['timestamp'], pepper['mid_price'].ffill().bfill(), 1)
            p = np.poly1d(z)
            axes[i].plot(pepper['timestamp'], p(pepper['timestamp']), "r--", alpha=0.8, label='Growth Trend')
            
            axes[i].set_title(f'Intarian Pepper Root Analysis: Day {day}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Price')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.2)
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    plt.xlabel('Timestamp')
    plt.tight_layout()
    plt.show()

# Run
plot_interpolated_pepper_days()

```


    
![png](round1_analysis_files/round1_analysis_11_0.png)
    



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pepper_cleaned():
    days = [-2, -1, 0]
    fig, axes = plt.subplots(len(days), 1, figsize=(15, 12), sharex=True)
    
    for i, day in enumerate(days):
        filename = f'prices_round_1_day_{day}.csv'
        try:
            df_day = pd.read_csv(filename, sep=';')
            pepper = df_day[df_day['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
            
            # 1. Remove zero or missing price rows
            # Added zero-value filter
            pepper = pepper[pepper['mid_price'] > 0]
            
            # 2. Visualization
            # Connect with solid line interpolation; mark actual data points
            axes[i].plot(pepper['timestamp'], pepper['mid_price'], 
                        linestyle='-', color='#3498db', alpha=0.6, label='Cleaned Path')
            
            axes[i].scatter(pepper['timestamp'], pepper['mid_price'], 
                           s=5, color='#2980b9', alpha=0.3, label='Actual Data')
            
            # 3. Elegant trend line
            z = np.polyfit(pepper['timestamp'], pepper['mid_price'], 1)
            p = np.poly1d(z)
            axes[i].plot(pepper['timestamp'], p(pepper['timestamp']), "r--", alpha=0.6, label='Predicted Trend')
            
            axes[i].set_title(f'Day {day}: Pepper Root (Filtered Zeroes)', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Price')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.1)
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    plt.xlabel('Timestamp')
    plt.tight_layout()
    plt.show()

# Run
plot_pepper_cleaned()

```


    
![png](round1_analysis_files/round1_analysis_12_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_in_intervals(day=0, num_intervals=10):
    # 1. Load data for specified day (default: Day 0)
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # Compute micro-price (for detailed analysis)
        pepper['micro_price'] = (pepper['bid_price_1'] * pepper['ask_volume_1'] + 
                                pepper['ask_price_1'] * pepper['bid_volume_1']) / \
                                (pepper['bid_volume_1'] + pepper['ask_volume_1'])

        # 2. Divide into time intervals
        interval_len = 1000000 // num_intervals
        
        # Create 10 subplot panels
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            # Filter data for this interval
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # Visualize mid-price and micro-price (intent)
                axes[i].plot(subset['timestamp'], subset['mid_price'], label='Mid Price', color='#7f8c8d', alpha=0.6)
                axes[i].plot(subset['timestamp'], subset['micro_price'], label='Micro Price', color='#e67e22', alpha=0.8)
                
                axes[i].set_title(f'Day {day} - Interval {i+1} ({start_ts} to {end_ts})', fontsize=12, fontweight='bold')
                axes[i].legend(loc='upper left')
                axes[i].grid(True, alpha=0.2)
            else:
                axes[i].set_title(f'Day {day} - Interval {i+1} (No Data)', color='red')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Run: detailed Day 0 analysis in 10 intervals
plot_pepper_in_intervals(day=0)

```


    
![png](round1_analysis_files/round1_analysis_13_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_ultra_detailed(day=0, num_intervals=50):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # Compute micro-price
        pepper['micro_price'] = (pepper['bid_price_1'] * pepper['ask_volume_1'] + 
                                pepper['ask_price_1'] * pepper['bid_volume_1']) / \
                                (pepper['bid_volume_1'] + pepper['ask_volume_1'])

        # 2. Divide into 20,000-timestamp intervals
        interval_len = 1000000 // num_intervals
        
        # Create 50 subplot panels
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                axes[i].plot(subset['timestamp'], subset['mid_price'], label='Mid', color='#7f8c8d', alpha=0.5)
                axes[i].plot(subset['timestamp'], subset['micro_price'], label='Micro', color='#e67e22', alpha=0.9)
                
                # Show interval range as title only (readability)
                axes[i].set_title(f'[{i+1}] {start_ts} ~ {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
            else:
                axes[i].text(0.5, 0.5, 'Empty', ha='center', va='center', alpha=0.2)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Run: fine-grained Day 0 analysis in 50 intervals
plot_pepper_ultra_detailed(day=0, num_intervals=50)

```


    
![png](round1_analysis_files/round1_analysis_14_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_full_detail(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load price and trade data
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        # Extract PEPPER ROOT data only
        p_pepper = prices[prices['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        t_pepper = trades[trades['symbol'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 2. Divide into time intervals
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            # Filter data for this interval
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = t_pepper[(t_pepper['timestamp'] >= start_ts) & (t_pepper['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # Best bid / ask (current LOB state)
                # Using 'step' style accurately shows quote changes at each tick.
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], label='Ask', color='#e74c3c', alpha=0.4, where='post')
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], label='Bid', color='#2ecc71', alpha=0.4, where='post')
                
                # Simple mid-price (trend reference)
                axes[i].plot(p_sub['timestamp'], p_sub['mid_price'], color='#bdc3c7', linestyle=':', alpha=0.5)
                
                # Trade (actual fill)
                if not t_sub.empty:
                    axes[i].scatter(t_sub['timestamp'], t_sub['price'], 
                                   s=15, color='black', marker='x', label='Trade', zorder=5)
                
                axes[i].set_title(f'[{i+1}] {start_ts} ~ {end_ts}', fontsize=10, loc='left')
                axes[i].legend(loc='upper right', fontsize=8)
                axes[i].grid(True, alpha=0.1)
                
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Run
plot_pepper_full_detail(day=0)

```


    
![png](round1_analysis_files/round1_analysis_15_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_trade_types(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load data and filter (ignore zero values)
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        # Extract PEPPER and remove zero-price rows
        p_pepper = prices[(prices['product'] == 'INTARIAN_PEPPER_ROOT') & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == 'INTARIAN_PEPPER_ROOT') & (trades['price'] > 0)].copy()
        
        # 2. Merge asof to classify fill side (buy vs. sell)
        # Need the prevailing bid/ask at each fill timestamp to classify direction.
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')
        
        combined_trades = pd.merge_asof(t_pepper, 
                                        p_pepper[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')
        
        # Classify fill: near ask → Buy (red); near bid → Sell (blue)
        def classify_trade(row):
            if row['price'] >= row['ask_price_1']: return 'red'  # Buy
            if row['price'] <= row['bid_price_1']: return 'blue' # Sell
            # If mid: classify toward closer side
            if abs(row['price'] - row['ask_price_1']) < abs(row['price'] - row['bid_price_1']):
                return 'red'
            return 'blue'

        combined_trades['color'] = combined_trades.apply(classify_trade, axis=1)

        # 3. Interval visualization
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # Bid/ask quote lines
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#e74c3c', alpha=0.2, where='post')
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], color='#2ecc71', alpha=0.2, where='post')
                
                # Fill markers (buy/sell classified)
                if not t_sub.empty:
                    # Buy fills (red)
                    buys = t_sub[t_sub['color'] == 'red']
                    axes[i].scatter(buys['timestamp'], buys['price'], s=30, color='red', marker='^', label='Buy Trade', zorder=5)
                    
                    # Sell fills (blue)
                    sells = t_sub[t_sub['color'] == 'blue']
                    axes[i].scatter(sells['timestamp'], sells['price'], s=30, color='blue', marker='v', label='Sell Trade', zorder=5)
                
                axes[i].set_title(f'[{i+1}] {start_ts} ~ {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Run
plot_pepper_trade_types(day=0)

```


    
![png](round1_analysis_files/round1_analysis_16_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_deep_book(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load data and filter
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        p_pepper = prices[(prices['product'] == 'INTARIAN_PEPPER_ROOT') & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == 'INTARIAN_PEPPER_ROOT') & (trades['price'] > 0)].copy()
        
        # Merge asof for fill classification
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')
        combined_trades = pd.merge_asof(t_pepper, 
                                        p_pepper[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')

        combined_trades['color'] = combined_trades.apply(
            lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
            axis=1
        )

        # 2. Interval visualization (including deep LOB levels)
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 5 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # --- Ask quotes (red) ---
                # Level 3 (deepest)
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_3'], color='#ff0000', alpha=0.1, where='post', label='Ask L3')
                # Level 2
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_2'], color='#ff0000', alpha=0.2, where='post', label='Ask L2')
                # Level 1 (best)
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#ff0000', alpha=0.5, where='post', label='Ask L1', linewidth=1.5)
                
                # --- Bid quotes (green) ---
                # Level 1 (best)
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], color='#00aa00', alpha=0.5, where='post', label='Bid L1', linewidth=1.5)
                # Level 2
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_2'], color='#00aa00', alpha=0.2, where='post', label='Bid L2')
                # Level 3 (deepest)
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_3'], color='#00aa00', alpha=0.1, where='post', label='Bid L3')
                
                # Fill markers
                if not t_sub.empty:
                    buys = t_sub[t_sub['color'] == 'red']
                    axes[i].scatter(buys['timestamp'], buys['price'], s=40, color='red', marker='^', label='Buy', zorder=10)
                    sells = t_sub[t_sub['color'] == 'blue']
                    axes[i].scatter(sells['timestamp'], sells['price'], s=40, color='blue', marker='v', label='Sell', zorder=10)
                
                axes[i].set_title(f'[{i+1}] {start_ts} ~ {end_ts} (L1-L3 Depth)', fontsize=10, loc='left')
                axes[i].legend(loc='upper right', fontsize=8, ncol=2)
                axes[i].grid(True, alpha=0.05)
                
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")

# Run
plot_pepper_deep_book(day=0)

```


    
![png](round1_analysis_files/round1_analysis_17_0.png)
    


---
## Section 3: ASH_COATED_OSMIUM — Mean-Reversion Market Making

Cells 15–18: Full multi-day analysis for OSMIUM. Price oscillates around 10,000 in-sample but drifts downward out-of-sample.

**Post-mortem:** The instantaneous mid-price anchor chased the drift, producing a sequence of buys at falling prices.  
**What would have worked better:** A slower anchor — VWAP or a longer rolling mid — would have been less reactive to short-term drift.



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_asset_full_analysis(product_name='ASH_COATED_OSMIUM', day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load data and filter target asset
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        p_sub_all = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
        t_sub_all = trades[(trades['symbol'] == product_name) & (trades['price'] > 0)].copy()
        
        # Merge asof for fill classification
        p_sub_all = p_sub_all.sort_values('timestamp')
        t_sub_all = t_sub_all.sort_values('timestamp')
        combined_trades = pd.merge_asof(t_sub_all, 
                                        p_sub_all[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')

        # Classify fill direction (buy/sell color)
        combined_trades['color'] = combined_trades.apply(
            lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
            axis=1
        )

        # 2. Visualize in 50 intervals
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 5 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_seg = p_sub_all[(p_sub_all['timestamp'] >= start_ts) & (p_sub_all['timestamp'] < end_ts)]
            t_seg = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_seg.empty:
                # --- Ask quotes (red) ---
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_3'], color='#ff0000', alpha=0.1, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_2'], color='#ff0000', alpha=0.2, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_1'], color='#ff0000', alpha=0.5, where='post', linewidth=1.5, label='Ask L1')
                
                # --- Bid quotes (green) ---
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_1'], color='#00aa00', alpha=0.5, where='post', linewidth=1.5, label='Bid L1')
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_2'], color='#00aa00', alpha=0.2, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_3'], color='#00aa00', alpha=0.1, where='post')
                
                # Fill markers (triangle style)
                if not t_seg.empty:
                    buys = t_seg[t_seg['color'] == 'red']
                    axes[i].scatter(buys['timestamp'], buys['price'], s=45, color='red', marker='^', label='Buy Trade', zorder=10)
                    sells = t_seg[t_seg['color'] == 'blue']
                    axes[i].scatter(sells['timestamp'], sells['price'], s=45, color='blue', marker='v', label='Sell Trade', zorder=10)
                
                axes[i].set_title(f'{product_name} [{i+1}] {start_ts}~{end_ts}', fontsize=10, loc='left')
                axes[i].legend(loc='upper right', fontsize=8, ncol=2)
                axes[i].grid(True, alpha=0.05)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error analyzing {product_name}: {e}")

# Run OSMIUM analysis
plot_asset_full_analysis(product_name='ASH_COATED_OSMIUM', day=0)

# To re-run PEPPER analysis instead: 
# plot_asset_full_analysis(product_name='INTARIAN_PEPPER_ROOT', day=0)

```


    
![png](round1_analysis_files/round1_analysis_19_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_spread_timeseries(day=0):
    price_file = f'prices_round_1_day_{day}.csv'
    
    try:
        # 1. Load data and compute spread
        df = pd.read_csv(price_file, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # Spread computation: Ask L1 - Bid L1
        pepper['spread'] = pepper['ask_price_1'] - pepper['bid_price_1']
        
        # 2. Visualization
        plt.figure(figsize=(15, 6))
        
        # Raw spread (light color)
        plt.plot(pepper['timestamp'], pepper['spread'], color='#3498db', alpha=0.3, label='Raw Spread')
        
        # Spread moving average (trend check)
        pepper['spread_sma'] = pepper['spread'].rolling(window=100).mean()
        plt.plot(pepper['timestamp'], pepper['spread_sma'], color='#2c3e50', linewidth=2, label='100-tick Moving Average')
        
        plt.title(f'Intarian Pepper Root: Bid-Ask Spread Time Series (Day {day})', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Spread Amount')
        
        # Add summary statistics
        mean_spread = pepper['spread'].mean()
        plt.axhline(mean_spread, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_spread:.2f}')
        
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
        
        # Print basic statistics
        print(f"--- Spread Statistics for Day {day} ---")
        print(f"Average Spread: {mean_spread:.2f}")
        print(f"Max Spread: {pepper['spread'].max()}")
        print(f"Min Spread: {pepper['spread'].min()}")
        print(f"Standard Deviation: {pepper['spread'].std():.2f}")

    except Exception as e:
        print(f"Error plotting spread: {e}")

# Run
plot_pepper_spread_timeseries(day=0)

```


    
![png](round1_analysis_files/round1_analysis_20_0.png)
    


    --- Spread Statistics for Day 0 ---
    Average Spread: 14.13
    Max Spread: 21.0
    Min Spread: 2.0
    Standard Deviation: 2.67
    


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_pepper_spread_trend():
    days = [-2, -1, 0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    daily_stats = []
    
    for i, day in enumerate(days):
        filename = f'prices_round_1_day_{day}.csv'
        try:
            df = pd.read_csv(filename, sep=';')
            pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
            pepper['spread'] = pepper['ask_price_1'] - pepper['bid_price_1']
            
            # Drop missing and zero values
            pepper = pepper[pepper['spread'] > 0]
            
            # 1. Visualization (raw + trend line)
            axes[i].plot(pepper['timestamp'], pepper['spread'], color='#3498db', alpha=0.2)
            
            # Moving average (overall trend)
            sma = pepper['spread'].rolling(window=500).mean()
            axes[i].plot(pepper['timestamp'], sma, color='#e74c3c', linewidth=2, label='Trend')
            
            # Linear regression to detect intraday spread trend
            z = np.polyfit(pepper['timestamp'], pepper['spread'], 1)
            p = np.poly1d(z)
            axes[i].plot(pepper['timestamp'], p(pepper['timestamp']), "k--", alpha=0.8, label='Linear fit')
            
            axes[i].set_title(f'Day {day} Spread', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Timestamp')
            if i == 0: axes[i].set_ylabel('Spread Amount')
            axes[i].grid(True, alpha=0.1)
            axes[i].legend(fontsize=8)
            
            # Store statistics
            daily_stats.append({
                'day': day,
                'mean': pepper['spread'].mean(),
                'slope': z[0] * 1000000 # change per 1M timestamps
            })
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    plt.suptitle('Intarian Pepper Root: Spread Trend Analysis (Over 3 Days)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Print statistics (cross-day comparison)
    print("\n--- Inter-Day Spread Trend Statistics ---")
    stats_df = pd.DataFrame(daily_stats)
    print(stats_df.to_string(index=False))
    
    if stats_df['mean'].is_monotonic_increasing:
        print("\n[Finding] Average spread is consistently *increasing* across days.")
    elif stats_df['mean'].is_monotonic_decreasing:
        print("\n[Finding] Average spread is consistently *decreasing* across days.")
    else:
        print("\n[Finding] No clear linear increase or decrease pattern in spread across days.")

# Run
analyze_pepper_spread_trend()

```


    
![png](round1_analysis_files/round1_analysis_21_0.png)
    


    
    --- Inter-Day Spread Trend Statistics ---
     day      mean    slope
      -2 11.994792 0.993379
      -1 13.012257 1.056795
       0 14.128715 0.928454
    
    [!] 결과: 일차별 평균 스프레드가 꾸준히 '증가'하고 있습니다.
    


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_asset_spread_trend(product_name='ASH_COATED_OSMIUM'):
    days = [-2, -1, 0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    daily_stats = []
    
    for i, day in enumerate(days):
        filename = f'prices_round_1_day_{day}.csv'
        try:
            df = pd.read_csv(filename, sep=';')
            asset_df = df[df['product'] == product_name].copy()
            asset_df['spread'] = asset_df['ask_price_1'] - asset_df['bid_price_1']
            
            # Drop missing and zero values
            asset_df = asset_df[asset_df['spread'] > 0]
            
            # 1. Visualization
            axes[i].plot(asset_df['timestamp'], asset_df['spread'], color='#16a085', alpha=0.2)
            
            # Moving average
            sma = asset_df['spread'].rolling(window=500).mean()
            axes[i].plot(asset_df['timestamp'], sma, color='#d35400', linewidth=2, label='Trend')
            
            # Linear regression
            z = np.polyfit(asset_df['timestamp'], asset_df['spread'], 1)
            p = np.poly1d(z)
            axes[i].plot(asset_df['timestamp'], p(asset_df['timestamp']), "k--", alpha=0.8, label='Linear fit')
            
            axes[i].set_title(f'Day {day} {product_name}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Timestamp')
            if i == 0: axes[i].set_ylabel('Spread Amount')
            axes[i].grid(True, alpha=0.1)
            axes[i].legend(fontsize=8)
            
            daily_stats.append({
                'day': day,
                'mean': asset_df['spread'].mean(),
                'slope': z[0] * 1000000 
            })
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    plt.suptitle(f'{product_name}: Spread Trend Analysis (Over 3 Days)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Print statistics
    stats_df = pd.DataFrame(daily_stats)
    print(f"\n--- {product_name} Inter-Day Spread Statistics ---")
    print(stats_df.to_string(index=False))

# Run OSMIUM analysis
analyze_asset_spread_trend('ASH_COATED_OSMIUM')

```


    
![png](round1_analysis_files/round1_analysis_22_0.png)
    


    
    --- ASH_COATED_OSMIUM Inter-Day Spread Statistics ---
     day      mean     slope
      -2 16.149777  0.006246
      -1 16.191328 -0.075977
       0 16.184467 -0.134077
    

---
## Section 4: Spread Analysis & Jump Detection

Cells 16–19: Time-series of bid-ask spread per asset. Spread trend analysis across days. True price jump detection (distinguishing fill-driven ticks from genuine price moves).



```python
import pandas as pd
import numpy as np

def analyze_pepper_true_jumps(day=0):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Replace 0.0 with NaN and forward-fill
        # Must remove zeros first to preserve the last valid value.
        pepper.loc[pepper['mid_price'] <= 0, 'mid_price'] = np.nan
        pepper['mid_price'] = pepper['mid_price'].ffill()
        
        # 2. Fill all timestamp gaps (0–999,900) using back-fill
        all_ts = np.arange(0, 1000000, 100)
        full_df = pd.DataFrame({'timestamp': all_ts})
        pepper = pd.merge(full_df, pepper, on='timestamp', how='left')
        
        # Apply back-fill to empty timestamp gaps
        pepper['mid_price'] = pepper['mid_price'].bfill()
        
        # 3. Jump analysis (difference from previous tick)
        pepper['price_diff'] = pepper['mid_price'].diff().abs()
        
        # Filter: non-zero price moves only
        jumps = pepper[pepper['price_diff'] > 0].copy()
        
        print(f"--- Pepper Price Jump Analysis (Day {day}, Cleaned Zeroes) ---")
        if not jumps.empty:
            print(f"Max jump magnitude: {jumps['price_diff'].max()}")
            print(f"Min jump magnitude: {jumps['price_diff'].min()}")
            
            # Show top 10 largest jumps
            top_10 = jumps.sort_values('price_diff', ascending=False).head(10)
            print("\n[Top 10 jump locations]")
            print(top_10[['timestamp', 'price_diff', 'mid_price']].to_string(index=False))
        else:
            print("No jump data to analyze.")
            
    except Exception as e:
        print(f"Error: {e}")

# Run: filter zeros on Day 0 then analyze jumps
analyze_pepper_true_jumps(day=0)

```

    --- Pepper Price Jump Analysis (Day 0, Cleaned Zeroes) ---
    최대 점프 폭: 18.0
    최소 점프 폭: 0.5
    
    [상위 10개 점프 지점]
     timestamp  price_diff  mid_price
        971500        18.0    12982.0
        565000        17.0    12555.0
        715500        17.0    12708.0
        595000        17.0    12588.0
        586200        17.0    12593.0
        359500        16.0    12369.0
        971600        16.0    12966.0
         20000        15.5    12030.0
        711900        15.0    12717.0
        706100        14.0    12699.0
    

---
## Section 5: Mid-Price Zoomed Analysis & Toxic Flow Screening

Cells 20–27: Progressive refinement of mid-price visualization across 30–100 intervals. Tests whether post-trade price movement is directional (toxic flow indicator).

**Toxic flow test:**  
For each trade, measure the price distribution over the subsequent 1–15 ticks.  
**Finding:** Under normal spread conditions, post-trade distributions showed no consistent directional bias — flow was not toxic in the conventional sense. However, trades placed during spread-break windows (bid ≥ ask) showed systematic adverse outcomes, consistent with toxic counterparties targeting anomalous spread windows.

> **Design observation:** The absence of FIFO queue priority creates a potential spread-break arbitrage. The competition blocked this through counterparty design — not market structure.



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_30_intervals(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # Filter PEPPER data excluding zero-price rows
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # Divide into ~33,333-timestamp intervals
        interval_len = 1000000 // num_intervals
        
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # Visualize price flow
                axes[i].plot(subset['timestamp'], subset['mid_price'], color='#2c3e50', linewidth=1.5)
                
                # Readability settings
                axes[i].set_title(f'Interval {i+1}: {start_ts} to {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                
                # Tighten Y-axis to data range (for granular change visibility)
                y_min, y_max = subset['mid_price'].min(), subset['mid_price'].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data (Zeroes Filtered)', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Run: detailed Day 0 mid-price analysis in 30 intervals
plot_pepper_mid_30_intervals(day=0)

```


    
![png](round1_analysis_files/round1_analysis_26_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_day_0_mid():
    # Day 0 file (competition day 3)
    filename = 'prices_round_1_day_0.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # Filter PEPPER data and remove noise
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        plt.figure(figsize=(15, 7))
        plt.plot(pepper['timestamp'], pepper['mid_price'], color='#2c3e50', linewidth=1, label='Pepper L3 Mid Price')
        
        # Add trend line
        import numpy as np
        z = np.polyfit(pepper['timestamp'], pepper['mid_price'], 1)
        p = np.poly1d(z)
        plt.plot(pepper['timestamp'], p(pepper['timestamp']), "r--", alpha=0.6, label='Overall Trend')
        
        plt.title('Intarian Pepper Root: L3 (Day 0) Mid Price Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
        
        print(f"L3 start price: {pepper['mid_price'].iloc[0]}")
        print(f"L3 end price: {pepper['mid_price'].iloc[-1]}")
        print(f"L3 mean price: {pepper['mid_price'].mean():.2f}")

    except FileNotFoundError:
        print(f"File {filename} not found. Check data filename.")

# Run
plot_day_0_mid()

```


    
![png](round1_analysis_files/round1_analysis_27_0.png)
    


    L3 시작 가격: 11998.5
    L3 종료 가격: 13000.0
    L3 평균 가격: 12500.17
    


```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_l3_mid_30_intervals(num_intervals=30):
    # L3 = Day 0 data
    filename = 'prices_round_1_day_0.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # Filter PEPPER and exclude zero-price rows
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # Price visualization
                axes[i].plot(subset['timestamp'], subset['mid_price'], color='#34495e', linewidth=1.5)
                
                # Interval label and auto Y-axis adjustment
                axes[i].set_title(f'L3 Interval {i+1}: {start_ts} to {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                
                y_min, y_max = subset['mid_price'].min(), subset['mid_price'].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'Empty Segment', ha='center', va='center', alpha=0.2)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Run: L3 data analysis in 30 intervals
plot_l3_mid_30_intervals(num_intervals=30)

```


    
![png](round1_analysis_files/round1_analysis_28_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l3_30_intervals(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Compute Level 3 mid price: (bid_3 + ask_3) / 2
        # Exclude intervals where data is zero
        pepper = pepper[(pepper['bid_price_3'] > 0) & (pepper['ask_price_3'] > 0)]
        pepper['mid_l3'] = (pepper['bid_price_3'] + pepper['ask_price_3']) / 2.0
        
        # 2. Divide into 30 intervals
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # Visualize Level 3 mid price
                axes[i].plot(subset['timestamp'], subset['mid_l3'], color='#8e44ad', linewidth=1.5, label='L3 Mid')
                
                # Uncomment below to compare with L1 mid price
                # axes[i].plot(subset['timestamp'], subset['mid_price'], color='#bdc3c7', alpha=0.5, label='L1 Mid')
                
                axes[i].set_title(f'Level 3 Mid Interval {i+1}: {start_ts} to {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                axes[i].legend(loc='upper right', fontsize=8)
                
                # Adjust Y-axis range
                y_min, y_max = subset['mid_l3'].min(), subset['mid_l3'].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No L3 Data', ha='center', va='center', alpha=0.2)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Run: PEPPER Level 3 mid-price analysis in 30 intervals
plot_pepper_mid_l3_30_intervals(day=0)

```


    
![png](round1_analysis_files/round1_analysis_29_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pepper_mid_l3_robust(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # Extract PEPPER data
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Compute Level 3 prices (cast to numeric, extract valid values)
        # Exclude zeros and NaN — keep only rows with valid L3 data.
        pepper['bid_price_3'] = pd.to_numeric(pepper['bid_price_3'], errors='coerce')
        pepper['ask_price_3'] = pd.to_numeric(pepper['ask_price_3'], errors='coerce')
        
        # Filter to find intervals with at least one L3 data point
        pepper_l3 = pepper.dropna(subset=['bid_price_3', 'ask_price_3']).copy()
        pepper_l3 = pepper_l3[(pepper_l3['bid_price_3'] > 0) & (pepper_l3['ask_price_3'] > 0)]
        
        if pepper_l3.empty:
            print(f"⚠️ Warning: {filename} has no Level 3 quote data for PEPPER!")
            # Fall back to Level 1 data
            print("Showing Level 1 mid price instead.")
            pepper_l3 = pepper[(pepper['mid_price'] > 0)].copy()
            target_col = 'mid_price'
        else:
            pepper_l3['mid_l3'] = (pepper_l3['bid_price_3'] + pepper_l3['ask_price_3']) / 2.0
            target_col = 'mid_l3'

        # 2. Divide into 30 intervals
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper_l3[(pepper_l3['timestamp'] >= start_ts) & (pepper_l3['timestamp'] < end_ts)]
            
            if not subset.empty:
                axes[i].plot(subset['timestamp'], subset[target_col], color='#8e44ad', linewidth=1.5)
                axes[i].set_title(f'Interval {i+1}: {start_ts} to {end_ts} ({target_col})', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                
                # Auto-optimize Y-axis range
                y_min, y_max = subset[target_col].min(), subset[target_col].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 2.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data in this range', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")

# Run
plot_pepper_mid_l3_robust(day=0)

```

    ⚠️ 경고: prices_round_1_day_0.csv 파일에 Pepper의 Level 3 호가 데이터가 하나도 없습니다!
    대신 Level 1 중간가(Mid Price)를 확인합니다.
    


    
![png](round1_analysis_files/round1_analysis_30_1.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l2_30_intervals(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # Extract PEPPER data
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Convert Level 2 prices to numeric (handle NaN and zeros)
        pepper['bid_price_2'] = pd.to_numeric(pepper['bid_price_2'], errors='coerce')
        pepper['ask_price_2'] = pd.to_numeric(pepper['ask_price_2'], errors='coerce')
        
        # Extract rows where L2 data exists
        pepper_l2 = pepper.dropna(subset=['bid_price_2', 'ask_price_2']).copy()
        pepper_l2 = pepper_l2[(pepper_l2['bid_price_2'] > 0) & (pepper_l2['ask_price_2'] > 0)]
        
        if pepper_l2.empty:
            print(f"⚠️ {filename}has insufficient Level 2 data. Falling back to Level 1 mid price.")
            pepper_l2 = pepper[pepper['mid_price'] > 0].copy()
            pepper_l2['target_price'] = pepper_l2['mid_price']
            tag = "Level 1 Mid"
        else:
            pepper_l2['target_price'] = (pepper_l2['bid_price_2'] + pepper_l2['ask_price_2']) / 2.0
            tag = "Level 2 Mid"

        # 2. Visualize in 30 intervals
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper_l2[(pepper_l2['timestamp'] >= start_ts) & (pepper_l2['timestamp'] < end_ts)]
            
            if not subset.empty:
                axes[i].plot(subset['timestamp'], subset['target_price'], color='#16a085', linewidth=1.5)
                axes[i].set_title(f'Interval {i+1}: {start_ts} to {end_ts} ({tag})', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                
                # Auto Y-axis scaling
                y_min, y_max = subset['target_price'].min(), subset['target_price'].max()
                margin = (y_max - y_min) * 0.2 if y_max != y_min else 2.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data (Interval Empty)', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Run: Level 2 mid-price analysis in 30 intervals
plot_pepper_mid_l2_30_intervals(day=0)

```


    
![png](round1_analysis_files/round1_analysis_31_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l2_ultra_detailed(day=0, num_intervals=100):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # Level 2 price quantification
        pepper['bid_price_2'] = pd.to_numeric(pepper['bid_price_2'], errors='coerce')
        pepper['ask_price_2'] = pd.to_numeric(pepper['ask_price_2'], errors='coerce')
        
        # Extract valid data (L2-prioritized rows)
        pepper_l2 = pepper.dropna(subset=['bid_price_2', 'ask_price_2']).copy()
        pepper_l2 = pepper_l2[(pepper_l2['bid_price_2'] > 0) & (pepper_l2['ask_price_2'] > 0)]
        
        if pepper_l2.empty:
            pepper_l2 = pepper[pepper['mid_price'] > 0].copy()
            pepper_l2['target_price'] = pepper_l2['mid_price']
            tag = "L1 Mid"
        else:
            pepper_l2['target_price'] = (pepper_l2['bid_price_2'] + pepper_l2['ask_price_2']) / 2.0
            tag = "L2 Mid"

        # Visualize in 100 intervals (compact layout)
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 2.5 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper_l2[(pepper_l2['timestamp'] >= start_ts) & (pepper_l2['timestamp'] < end_ts)]
            
            if not subset.empty:
                axes[i].plot(subset['timestamp'], subset['target_price'], color='#27ae60', linewidth=1.2)
                axes[i].set_title(f'[{i+1}] {start_ts}~{end_ts} ({tag})', fontsize=9, loc='left', pad=2)
                axes[i].grid(True, alpha=0.1)
                
                y_min, y_max = subset['target_price'].min(), subset['target_price'].max()
                margin = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].set_title(f'[{i+1}] {start_ts} (Empty)', fontsize=9, loc='left', color='red')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Run: ultra-detailed Level 2 analysis in 100 intervals
plot_pepper_mid_l2_ultra_detailed(day=0, num_intervals=100)

```


    
![png](round1_analysis_files/round1_analysis_32_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_asset_3day_detailed(product_name='ASH_COATED_OSMIUM', num_intervals_per_day=10):
    days = [-2, -1, 0]
    total_intervals = len(days) * num_intervals_per_day
    
    # Create 30 subplots (3 days × 10 intervals)
    fig, axes = plt.subplots(total_intervals, 1, figsize=(15, 5 * total_intervals))
    
    interval_idx = 0
    for day in days:
        price_file = f'prices_round_1_day_{day}.csv'
        trade_file = f'trades_round_1_day_{day}.csv'
        
        try:
            # 1. Load data and filter
            prices = pd.read_csv(price_file, sep=';')
            trades = pd.read_csv(trade_file, sep=';')
            
            p_asset = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
            t_asset = trades[(trades['symbol'] == product_name) & (trades['price'] > 0)].copy()
            
            # Merge asof for buy/sell fill classification
            p_asset = p_asset.sort_values('timestamp')
            t_asset = t_asset.sort_values('timestamp')
            combined_trades = pd.merge_asof(t_asset, p_asset[['timestamp', 'bid_price_1', 'ask_price_1']], on='timestamp')
            
            combined_trades['color'] = combined_trades.apply(
                lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
                axis=1
            )

            # 2. Divide each day into 10 intervals for visualization
            interval_len = 1000000 // num_intervals_per_day
            for i in range(num_intervals_per_day):
                start_ts = i * interval_len
                end_ts = (i + 1) * interval_len
                
                p_seg = p_asset[(p_asset['timestamp'] >= start_ts) & (p_asset['timestamp'] < end_ts)]
                t_seg = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
                
                ax = axes[interval_idx]
                if not p_seg.empty:
                    # Ask quotes (L1–L3)
                    ax.step(p_seg['timestamp'], p_seg['ask_price_3'], color='red', alpha=0.1, where='post')
                    ax.step(p_seg['timestamp'], p_seg['ask_price_2'], color='red', alpha=0.2, where='post')
                    ax.step(p_seg['timestamp'], p_seg['ask_price_1'], color='red', alpha=0.5, where='post', linewidth=1.2, label='Ask L1')
                    
                    # Bid quotes (L1–L3)
                    ax.step(p_seg['timestamp'], p_seg['bid_price_1'], color='green', alpha=0.5, where='post', linewidth=1.2, label='Bid L1')
                    ax.step(p_seg['timestamp'], p_seg['bid_price_2'], color='green', alpha=0.2, where='post')
                    ax.step(p_seg['timestamp'], p_seg['bid_price_3'], color='green', alpha=0.1, where='post')
                    
                    # Fill markers
                    if not t_seg.empty:
                        buys = t_seg[t_seg['color'] == 'red']
                        ax.scatter(buys['timestamp'], buys['price'], s=40, color='red', marker='^', zorder=5)
                        sells = t_seg[t_seg['color'] == 'blue']
                        ax.scatter(sells['timestamp'], sells['price'], s=40, color='blue', marker='v', zorder=5)
                    
                    ax.set_title(f'Day {day} | Interval {i+1} ({start_ts}~{end_ts})', fontsize=10, loc='left')
                    ax.grid(True, alpha=0.05)
                else:
                    ax.text(0.5, 0.5, f'Day {day} Interval {i+1} No Data', ha='center')
                
                interval_idx += 1
                
        except Exception as e:
            print(f"Error Day {day}: {e}")
            for _ in range(num_intervals_per_day):
                axes[interval_idx].text(0.5, 0.5, f"Error Loading Day {day}", ha='center')
                interval_idx += 1

    plt.tight_layout()
    plt.show()

# Run: OSMIUM 3-day analysis
plot_asset_3day_detailed(product_name='ASH_COATED_OSMIUM', num_intervals_per_day=10)

# Uncomment below to run PEPPER analysis instead
# plot_asset_3day_detailed(product_name='INTARIAN_PEPPER_ROOT', num_intervals_per_day=10)

```


    
![png](round1_analysis_files/round1_analysis_33_0.png)
    

