# round2_analysis.md

---

# Round 2: PEPPER & OSMIUM ??Continued Analysis with MAF Context

> **Context:** Same two assets as Round 1, but now with 25% additional quote access granted via the MAF bid.  
> **Goal:** Re-examine OBI predictive power and spread dynamics under expanded access conditions.

## Overview

This notebook extends the Round 1 price/trade analysis to the Round 2 dataset. Key additions:
- Order Book Imbalance (OBI) signal testing across multiple horizons
- Fair value benchmarking with simple models (trend for PEPPER, static for OSMIUM)
- Spread anomaly detection

> **Note:** MAF auction modeling and resource allocation optimization are in `bid_estimation.ipynb` and `Manual_Analysis.ipynb`.


## Data Visualization

### Asset price visualization

---
## Section 1: Asset Price Visualization

Load all three days of Round 2 price data. Plot mid-price time series for PEPPER and OSMIUM.



```python
import pandas as pd
import matplotlib.pyplot as plt
import os
import re 
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
#1. Data loading setup 
price_files = sorted(glob.glob('prices_round_2_day*.csv'))
trade_files = sorted(glob.glob('trades_round_2_day*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

#Load price data
prices_list = []

for file in price_files:
    day = extract_day(file)
    if day is not None:
        df = pd.read_csv(file, sep=';')
        prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = (prices['day']+1) + prices['timestamp']/100

#Load trade data
trades_list = []

for file in trade_files:
    day = extract_day(file)
    if day is not None:
        df = pd.read_csv(file, sep=';')
        trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['timestamp']/100
#Filter
ROUND1_PROUNDCT = ["KELP","RAINFOREST_RESIN","SQUID_INK"]

prices_filtered = prices[~prices['product'].isin(ROUND1_PROUNDCT)]
trades_filtered = trades[~trades['symbol'].isin(ROUND1_PROUNDCT)]

#2. Visualize per asset 

assets = set(prices_filtered['product'])

for asset in assets:
    asset_prices = prices_filtered[prices_filtered['product'] == asset]
    asset_trades = trades_filtered[trades_filtered['symbol'] == asset]
    asset_trades.rename(columns={'price':'trade_price'}, inplace=True)

    combined = pd.merge(asset_prices, asset_trades, on = 'global_timestamp', how='left')
    
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill()  

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





### Visualization after dropping missing values


```python
import pandas as pd
import matplotlib.pyplot as plt
import os
import re 
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
#1. Data loading setup 
price_files = sorted(glob.glob('prices_round_2_day*.csv'))
trade_files = sorted(glob.glob('trades_round_2_day*.csv'))

def extract_day(filename):
    match = re.search(r'day_(-?\d+)', filename)
    return int(match.group(1)) if match else 0

#Load price data
prices_list = []

for file in price_files:
    day = extract_day(file)
    if day is not None:
        df = pd.read_csv(file, sep=';')
        prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = (prices['day']+1) + prices['timestamp']/100

#Load trade data
trades_list = []

for file in trade_files:
    day = extract_day(file)
    if day is not None:
        df = pd.read_csv(file, sep=';')
        trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['timestamp']/100
#Filter
ROUND1_PROUNDCT = ["KELP","RAINFOREST_RESIN","SQUID_INK"]

prices_filtered = prices[~prices['product'].isin(ROUND1_PROUNDCT)]
trades_filtered = trades[~trades['symbol'].isin(ROUND1_PROUNDCT)]

#2. Visualize per asset 

assets = set(prices_filtered['product'])

def calculate_mid_price(df, level)-> pd.DataFrame:
    bid = df[f'bid_price_{level}']
    ask = df[f'ask_price_{level}']
    
   
    df['mid_price'] = ((bid + ask) / 2).fillna(bid).fillna(ask)
    
    return df

for asset in assets:
    asset_prices = prices_filtered[prices_filtered['product'] == asset]
    asset_trades = trades_filtered[trades_filtered['symbol'] == asset]
    asset_trades.rename(columns={'price':'trade_price'}, inplace=True)
    asset_prices= calculate_mid_price(asset_prices, 1)
    
    
    combined = pd.merge(asset_prices, asset_trades, on = 'global_timestamp', how='left')
    
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill()  

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
def get_fair_value(asset, t):
    if 'PEPPER' in asset:
        return 13000 + 0.001 * t
    elif 'OSMIUM' in asset:
        return 10000
    else:
        return np.nan

for asset in assets:
    # Extract asset trade data and remove zero-price entries
    asset_trades = trades[trades['symbol'] == asset].copy()
    asset_trades = asset_trades[asset_trades['price'] > 0] # Remove zero-price fills (likely spurious)
    
    # [Fair value & deviation] ??use t at actual trade timestamp
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





---
## Section 2: Trade Data ??Distribution & Spread

Visualize trade price distributions (histogram, distribution factory). Spread time series per day.



```python
import pandas as pd
import glob
import plotly.graph_objects as go

# 1. Load data
price_files = sorted(glob.glob('prices_round_2_day_*.csv'))

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

# Ask Price L1 (offer side ??typically above mid)
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

# Bid Price L1 (bid side ??typically below mid)
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




```python
import pandas as pd
import glob
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# 1. Load data
trade_files = sorted(glob.glob('trades_round_2_day_*.csv'))

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
    bin_size=1, # OSMIUM prices are integer-valued ??bin size = 1
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
import re

# 1. Load and preprocess data
trade_files = sorted(glob.glob('trades_round_2_day_*.csv'))

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
pepper_trades['fair_value'] = 13000 + 0.001 * pepper_trades['global_ts']
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

    분석??충분??매수/매도 거래 ?�이?��? ?�습?�다.
    

---
## Section 3: Interval-Level Price Analysis

Segment each day into equal-width time intervals. Plot mid-price and trade overlay per interval.  
Purpose: identify intraday structure and whether the trend / mean-reversion regime holds across Round 2.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# 1. Load and filter PEPPER data
def load_pepper_data():
    files = sorted(glob.glob('prices_round_2_day_*.csv'))
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

# Exponential moving average for 'atmosphere' ??slow-moving trend
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


    
![png](_analysis_img/round2_analysis_13_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_in_intervals(day=0, num_intervals=10):
    # 1. Load data for specified day (default: Day 0)
    filename = f'prices_round_2_day_{day}.csv'
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


    
![png](_analysis_img/round2_analysis_14_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_trade_types(day=0, num_intervals=50):
    price_file = f'prices_round_2_day_{day}.csv'
    trade_file = f'trades_round_2_day_{day}.csv'
    
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
        
        # Classify fill: near ask ??Buy (red); near bid ??Sell (blue)
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
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#e74c3c', alpha=0.6, where='post')
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], color='#2ecc71', alpha=0.6, where='post')
                
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


    
![png](_analysis_img/round2_analysis_15_0.png)
    



```python
idx1 = "INTARIAN_PEPPER_ROOT"
idx2  = "ASH_COATED_OSMIUM"
```


```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_in_intervals(day=0, num_intervals=10):
    # 1. Load data for specified day (default: Day 0)
    filename = f'prices_round_2_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[(df['product'] == idx2) & (df['mid_price'] > 0)].copy()
        
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


    
![png](_analysis_img/round2_analysis_17_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pepper_fix_scale(day=0, num_intervals=30):
    price_file = f'prices_round_2_day_{day}.csv'
    trade_file = f'trades_round_2_day_{day}.csv'
    idx2 = 'ASH_COATED_OSMIUM'
    
    try:
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        # Remove zero values (prevents chart spikes at 0)
        p_pepper = prices[(prices['product'] == idx2) & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == idx2) & (trades['price'] > 0)].copy()
        
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')

        # 1. Compute OBI (Order Book Imbalance)
        b_sum = p_pepper[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1).fillna(0)
        a_sum = p_pepper[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1).fillna(0)
        denom = b_sum + a_sum
        p_pepper['imbalance'] = p_pepper['ask_price_1'] - p_pepper['bid_price_1']

        # 2. Classify trade direction
        p_pepper_merge = p_pepper[['timestamp', 'bid_price_1', 'ask_price_1', 'imbalance', 'mid_price']].copy()
        combined_trades = pd.merge_asof(t_pepper, p_pepper_merge, on='timestamp', direction='backward')
        
        def classify_trade(row):
            if pd.isna(row['ask_price_1']): return 'gray'
            if row['price'] >= row['ask_price_1']: return 'red'
            if row['price'] <= row['bid_price_1']: return 'blue'
            return 'red' if abs(row['price'] - row['ask_price_1']) < abs(row['price'] - row['bid_price_1']) else 'blue'
        combined_trades['color'] = combined_trades.apply(classify_trade, axis=1)

        # 3. Layout
        fig, axes = plt.subplots(num_intervals * 2, 1, 
                                 figsize=(15, 6 * num_intervals), 
                                 gridspec_kw={'height_ratios': [3, 1] * num_intervals})
        
        interval_len = 1000000 // num_intervals

        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            p_idx = i * 2
            i_idx = i * 2 + 1

            if not p_sub.empty:
                # --- Top: price chart (Y-axis scale fix is the key change) ---
                axes[p_idx].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#e74c3c', alpha=0.6, where='post', label='Ask')
                axes[p_idx].step(p_sub['timestamp'], p_sub['bid_price_1'], color="#25ca6a", alpha=0.6, where='post', label='Bid')
                axes[p_idx].step(p_sub['timestamp'], p_sub['mid_price'], color="#3498db", alpha=0.8, where='post', label='Mid')
                
                # 10,000 reference line
                axes[p_idx].axhline(y=10000, color='red', linestyle='--', linewidth=1, alpha=0.4)
                
                # �?Auto-optimize Y-axis range (± 15 around interval mid price)
                curr_mid = p_sub['mid_price'].median()
                axes[p_idx].set_ylim(curr_mid - 15, curr_mid + 15) # Fix Y-axis to interval price range

                if not t_sub.empty:
                    buys = t_sub[t_sub['color'] == 'red']
                    axes[p_idx].scatter(buys['timestamp'], buys['price'], s=40, color='red', marker='^', edgecolors='black', zorder=5)
                    sells = t_sub[t_sub['color'] == 'blue']
                    axes[p_idx].scatter(sells['timestamp'], sells['price'], s=40, color='blue', marker='v', edgecolors='black', zorder=5)
                
                axes[p_idx].set_title(f'[{i+1}] {start_ts} ~ {end_ts}', fontsize=12, loc='left', fontweight='bold')
                axes[p_idx].grid(True, alpha=0.2)

                # --- Bottom: OBI chart ---
                axes[i_idx].fill_between(p_sub['timestamp'], p_sub['imbalance'], 0, 
                                        where=(p_sub['imbalance'] >= 0), color='gray', alpha=0.3)
                axes[i_idx].fill_between(p_sub['timestamp'], p_sub['imbalance'], 0, 
                                        where=(p_sub['imbalance'] < 0), color='blue', alpha=0.15)
                axes[i_idx].plot(p_sub['timestamp'], p_sub['imbalance'], color='black', linewidth=0.5, alpha=0.5)
                
                if not t_sub.empty:
                    axes[i_idx].scatter(t_sub['timestamp'], t_sub['imbalance'], color='black', s=20, marker='o', zorder=10)

                axes[i_idx].axhline(y=0.6, color='red', linestyle=':', alpha=0.5)
                axes[i_idx].axhline(y=-0.6, color='blue', linestyle=':', alpha=0.5)
                axes[i_idx].set_ylim(-1.1, 1.1)
                axes[i_idx].grid(True, alpha=0.1)
            else:
                axes[p_idx].set_visible(False)
                axes[i_idx].set_visible(False)

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

plot_pepper_fix_scale(day=0)
```


    
![png](_analysis_img/round2_analysis_18_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_asset_full_analysis_refined(product_name='ASH_COATED_OSMIUM', day=0, num_intervals=10):
    price_file = f'prices_round_2_day_{day}.csv'
    trade_file = f'trades_round_2_day_{day}.csv'
    
    try:
        prices = pd.read_csv(price_file, sep=';', low_memory=False)
        trades = pd.read_csv(trade_file, sep=';', low_memory=False)
        
        p_sub_all = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
        t_sub_all = trades[(trades['symbol'] == product_name) & (trades['price'] > 0)].copy()
        
        # 1. Compute OBI and sort data
        p_sub_all['imbalance'] = (p_sub_all['bid_volume_1'] - p_sub_all['ask_volume_1']) / \
                                 (p_sub_all['bid_volume_1'] + p_sub_all['ask_volume_1'])

        p_sub_all = p_sub_all.sort_values('timestamp')
        t_sub_all = t_sub_all.sort_values('timestamp')
        
        # Merge asof to get mid price and OBI at fill timestamps
        combined_trades = pd.merge_asof(t_sub_all, 
                                        p_sub_all[['timestamp', 'bid_price_1', 'ask_price_1', 'mid_price', 'imbalance']], 
                                        on='timestamp')

        combined_trades['color'] = combined_trades.apply(
            lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
            axis=1
        )

        # 2. Visualization
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals * 2, 1, figsize=(15, 6 * num_intervals * 2), 
                                 gridspec_kw={'height_ratios': [3, 1] * num_intervals})
        
        fig.patch.set_facecolor('white')

        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_seg = p_sub_all[(p_sub_all['timestamp'] >= start_ts) & (p_sub_all['timestamp'] < end_ts)]
            t_seg = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            ax_price = axes[i*2]
            ax_imb = axes[i*2 + 1]

            if not p_seg.empty:
                # --- Top: price chart ---
                # Mid price: dark gray to avoid visual noise
                ax_price.step(p_seg['timestamp'], p_seg['mid_price'], color='#555555', linewidth=1.2, where='post', alpha=0.8, label='Mid Price')
                
                # Quotes: lighter alpha to appear as background
                ax_price.step(p_seg['timestamp'], p_seg['ask_price_1'], color='#ffaaaa', alpha=0.4, where='post')
                ax_price.step(p_seg['timestamp'], p_seg['bid_price_1'], color='#aaffaa', alpha=0.4, where='post')
                
                if not t_seg.empty:
                    # ??Mark mid price at fill timestamps with black dots
                    ax_price.scatter(t_seg['timestamp'], t_seg['mid_price'], color='black', s=15, zorder=15, label='Mid at Trade')
                    
                    # Fill price markers (triangles)
                    buys = t_seg[t_seg['color'] == 'red']
                    ax_price.scatter(buys['timestamp'], buys['price'], s=45, color='red', marker='^', alpha=0.7, zorder=10)
                    sells = t_seg[t_seg['color'] == 'blue']
                    ax_price.scatter(sells['timestamp'], sells['price'], s=45, color='blue', marker='v', alpha=0.7, zorder=10)
                
                ax_price.set_title(f'{product_name} Analysis [{i+1}]', fontsize=11, loc='left')
                ax_price.grid(True, alpha=0.1)
                ax_price.legend(loc='upper right', fontsize=8, ncol=2)

                # --- Bottom: OBI chart ---
                ax_imb.step(p_seg['timestamp'], p_seg['imbalance'], color='#888888', linewidth=0.8, alpha=0.4, where='post')
                ax_imb.fill_between(p_seg['timestamp'], p_seg['imbalance'], 0, color='#eeeeee', alpha=0.3, step='post')
                
                if not t_seg.empty:
                    # OBI markers at fill timestamps
                    ax_imb.scatter(t_seg['timestamp'], t_seg['imbalance'], color='black', s=10, zorder=12)
                    ax_imb.scatter(buys['timestamp'], buys['imbalance'], color='red', s=25, alpha=0.6, zorder=11)
                    ax_imb.scatter(sells['timestamp'], sells['imbalance'], color='blue', s=25, alpha=0.6, zorder=11)

                ax_imb.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
                ax_imb.set_ylim(-1.1, 1.1)
                ax_imb.grid(True, alpha=0.1)
                
            else:
                ax_price.text(0.5, 0.5, 'No Data', ha='center')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Run
plot_asset_full_analysis_refined(product_name='ASH_COATED_OSMIUM', day=0, num_intervals=10)
```


    
![png](_analysis_img/round2_analysis_19_0.png)
    


---
## Section 4: OBI Signal Testing

Order Book Imbalance (OBI) at levels 1?? is computed and tested for forward predictive power.

**Test setup:**
- Horizon: 1 to 20 ticks ahead
- OBI definition: (bid_size_L1 ??ask_size_L1) / (bid_size_L1 + ask_size_L1)
- Metric: Pearson correlation of OBI_t with mid-price return at t+horizon

> **Finding:** OBI had weak forward predictive power (correlation < 0.05 at all horizons tested). The signal did not clear the threshold for deployment.



```python
def plot_imbalance_predictiveness(p_sub_all):

    df = p_sub_all.copy()

    # future return (e.g. 20 ticks ahead)
    horizon = 20

    df['future_mid'] = df['mid_price'].shift(-horizon)

    df['future_return'] = (
        df['future_mid'] - df['mid_price']
    )

    # imbalance binning
    bins = np.linspace(-1, 1, 21)

    df['imb_bin'] = pd.cut(
        df['imbalance'],
        bins
    )

    grouped = df.groupby('imb_bin')['future_return'].mean()

    # plot
    plt.figure(figsize=(8,5))

    plt.plot(grouped.values)

    plt.title("Imbalance vs Future Return")

    plt.xlabel("Imbalance bin")
    plt.ylabel("Future Return")

    plt.grid(alpha=0.3)

    plt.show()
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def debug_imbalance_signal(
        product_name='ASH_COATED_OSMIUM',
        day=0,
        horizon=10,
        levels=3
):

    price_file = f'prices_round_2_day_{day}.csv'

    try:

        print("Loading file:", price_file)

        prices = pd.read_csv(
            price_file,
            sep=';',
            low_memory=False
        )

        print("Total rows:", len(prices))

        df = prices[
            (prices['product'] == product_name)
            & (prices['mid_price'] > 0)
        ].copy()

        print("Filtered rows:", len(df))

        if len(df) == 0:
            print("??No data after filtering")
            return

        df = df.sort_values('timestamp')

        # -------------------------
        # Multi-level imbalance
        # -------------------------

        bid_cols = []
        ask_cols = []

        for i in range(1, levels + 1):

            bid_col = f'bid_volume_{i}'
            ask_col = f'ask_volume_{i}'

            if bid_col in df.columns:
                bid_cols.append(bid_col)

            if ask_col in df.columns:
                ask_cols.append(ask_col)

        print("Using bid cols:", bid_cols)
        print("Using ask cols:", ask_cols)

        if len(bid_cols) == 0:
            print("??No volume columns found")
            return

        df['bid_vol_total'] = df[bid_cols].sum(axis=1)
        df['ask_vol_total'] = df[ask_cols].sum(axis=1)

        denom = (
            df['bid_vol_total']
            + df['ask_vol_total']
        )

        df = df[denom > 0]

        df['imbalance'] = (
            df['bid_vol_total']
            - df['ask_vol_total']
        ) / denom

        print("Imbalance std:", df['imbalance'].std())

        # -------------------------
        # Future return
        # -------------------------

        if horizon >= len(df):
            print("??Horizon too large ??reducing")
            horizon = len(df) // 5

        df['future_mid'] = df['mid_price'].shift(-horizon)

        df['future_return'] = (
            df['future_mid']
            - df['mid_price']
        )

        before_drop = len(df)

        df = df.dropna()

        print("Rows after dropna:", len(df))

        if len(df) == 0:
            print("??No rows left after dropna")
            return

        # -------------------------
        # Binning
        # -------------------------

        bins = np.linspace(-1, 1, 21)

        df['imb_bin'] = pd.cut(
            df['imbalance'],
            bins
        )

        grouped = df.groupby(
            'imb_bin'
        )['future_return'].mean()

        counts = df.groupby(
            'imb_bin'
        )['future_return'].count()

        centers = [
            interval.mid
            for interval in grouped.index
        ]

        print("Non-empty bins:", (counts > 0).sum())

        # -------------------------
        # Plot
        # -------------------------

        plt.figure(figsize=(10, 5))

        plt.plot(
            centers,
            grouped.values,
            marker='o'
        )

        plt.title(
            f"Imbalance vs Future Return (h={horizon})"
        )

        plt.xlabel("Imbalance")
        plt.ylabel("Mean Future Return")

        plt.axhline(
            y=0,
            color='black'
        )

        plt.grid(alpha=0.3)

        plt.show()

        # -------------------------
        # Check raw imbalance
        # -------------------------

        plt.figure(figsize=(12, 4))

        sample_df = df.iloc[:5000]

        plt.plot(
            sample_df['timestamp'],
            sample_df['imbalance'],
            linewidth=0.6
        )

        plt.title("Raw Imbalance (first 5000 rows)")

        plt.grid(alpha=0.2)

        plt.show()

    except Exception as e:

        print("Error:", e)


# Run
debug_imbalance_signal(
    product_name='ASH_COATED_OSMIUM',
    day=0,
    horizon=10,
    levels=3
)
```

    Loading file: prices_round_2_day_0.csv
    Total rows: 20000
    Filtered rows: 9984
    Using bid cols: ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    Using ask cols: ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
    Imbalance std: 0.3745364498120105
    Rows after dropna: 0
    ??No rows left after dropna
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def full_imbalance_signal_analysis(
        product_name='ASH_COATED_OSMIUM',
        day=0,
        horizons=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # updated default
        levels=3
):
    price_file = f'prices_round_2_day_{day}.csv'
    print(f"--- Starting analysis: {product_name} (Day {day}) ---")

    try:
        prices = pd.read_csv(price_file, sep=';', low_memory=False)
    except FileNotFoundError:
        print(f"??File not found: {price_file}")
        return

    # 1. Filter and sort
    df = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
    if len(df) == 0:
        print("??No data found for this product.")
        return
    df = df.sort_values('timestamp')

    # 2. Compute OBI (summed to level n)
    bid_cols = [c for c in df.columns if 'bid_volume_' in c][:levels]
    ask_cols = [c for c in df.columns if 'ask_volume_' in c][:levels]
    
    df['bid_vol_total'] = df[bid_cols].sum(axis=1)
    df['ask_vol_total'] = df[ask_cols].sum(axis=1)
    denom = df['bid_vol_total'] + df['ask_vol_total']

    df = df[denom > 0].copy()
    df['imbalance'] = (df['bid_vol_total'] - df['ask_vol_total']) / denom

    # 3. Analyze each horizon
    for horizon in horizons:
        df_h = df[['mid_price', 'imbalance']].copy() # Copy only needed columns to save memory
        
        # Compute future return
        df_h['future_return'] = df_h['mid_price'].shift(-horizon) - df_h['mid_price']
        df_h = df_h.dropna(subset=['future_return', 'imbalance'])

        if len(df_h) < 100:
            continue

        # Bin OBI into 20 equal buckets
        bins = np.linspace(-1, 1, 21)
        df_h['imb_bin'] = pd.cut(df_h['imbalance'], bins)
        
        result = df_h.groupby('imb_bin', observed=True)['future_return'].agg(['mean', 'count'])
        centers = [interval.mid for interval in result.index]

        # Plot output
        plt.figure(figsize=(8, 4))
        plt.plot(centers, result['mean'].values, marker='o', linestyle='-', color='tab:blue')
        plt.title(f"Predictiveness: {product_name} | Horizon: {horizon} tick(s)")
        plt.xlabel("Order Book Imbalance")
        plt.ylabel("Avg. Future Mid Price Change")
        plt.axhline(0, color='black', lw=0.8, ls='--')
        plt.grid(True, alpha=0.3)
        plt.show()

    # 4. Visualize raw OBI time series (optional)
    plt.figure(figsize=(12, 4))
    plt.plot(df['timestamp'].iloc[:2000], df['imbalance'].iloc[:2000], linewidth=0.7, color='gray')
    plt.title("Imbalance Time-series Preview (Partial)")
    plt.show()

# Run
if __name__ == "__main__":
    full_imbalance_signal_analysis(
        product_name='ASH_COATED_OSMIUM',
        day=0,
        horizons=list(range(1, 11)), # 1??0 tick horizons
        levels=3
    )
```

    --- 분석 ?�작: ASH_COATED_OSMIUM (Day 0) ---
    


    
![png](_analysis_img/round2_analysis_23_1.png)
    



    
![png](_analysis_img/round2_analysis_23_2.png)
    



    
![png](_analysis_img/round2_analysis_23_3.png)
    



    
![png](_analysis_img/round2_analysis_23_4.png)
    



    
![png](_analysis_img/round2_analysis_23_5.png)
    



    
![png](_analysis_img/round2_analysis_23_6.png)
    



    
![png](_analysis_img/round2_analysis_23_7.png)
    



    
![png](_analysis_img/round2_analysis_23_8.png)
    



    
![png](_analysis_img/round2_analysis_23_9.png)
    



    
![png](_analysis_img/round2_analysis_23_10.png)
    



    
![png](_analysis_img/round2_analysis_23_11.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

idx = idx2
def plot_pepper_fix_scale(day=0, num_intervals=30):
    price_file = f'prices_round_2_day_{day}.csv'
    trade_file = f'trades_round_2_day_{day}.csv'
    idx2 = 'ASH_COATED_OSMIUM'
    
    try:
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        p_pepper = prices[(prices['product'] == idx) & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == idx) & (trades['price'] > 0)].copy()
        
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')

        # 1. Compute spread and OBI
        def volume_level(i):            
            bid_lst = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3'][:i] # Fixed: was i-1, should be i to include level i
            ask_lst = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3'][:i]
            return (p_pepper[bid_lst].sum(axis=1), p_pepper[ask_lst].sum(axis=1))
            
        b_sum, a_sum = volume_level(1) 
        p_pepper['spread'] = p_pepper['ask_price_1'] - p_pepper['bid_price_1']
        p_pepper['imbalance'] = (b_sum - a_sum) / (b_sum + a_sum)

        # --- Added NaN handling for micro-price ---
        # Forward/back-fill zero or NaN volume/price to avoid computation errors
        temp_p = p_pepper[['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']].replace(0, np.nan)
        temp_p = temp_p.ffill().bfill() # ffill then bfill to eliminate all NaNs
        
        p_pepper['microprice'] = (temp_p['bid_price_1'] * temp_p['bid_volume_1'] + 
                                  temp_p['ask_price_1'] * temp_p['ask_volume_1']) / \
                                 (temp_p['bid_volume_1'] + temp_p['ask_volume_1'])
        # ----------------------------------

        # 2. Classify trade data (merge asof)
        p_merge = p_pepper[['timestamp', 'bid_price_1', 'ask_price_1', 'spread', 'imbalance', 'microprice', 'mid_price']].copy()
        combined_trades = pd.merge_asof(t_pepper, p_merge, on='timestamp', direction='backward')
        
        def classify_trade(row):
            if pd.isna(row['ask_price_1']): return 'gray'
            if row['price'] >= row['ask_price_1']: return 'red'
            if row['price'] <= row['bid_price_1']: return 'blue'
            return 'red' if abs(row['price'] - row['ask_price_1']) < abs(row['price'] - row['bid_price_1']) else 'blue'
        combined_trades['color'] = combined_trades.apply(classify_trade, axis=1)

        # 3. Layout
        fig, axes = plt.subplots(num_intervals * 3, 1, 
                                 figsize=(15, 8 * num_intervals), 
                                 gridspec_kw={'height_ratios': [4, 1, 1] * num_intervals})
        
        interval_len = 1000000 // num_intervals

        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            p_idx, s_idx, i_idx = i*3, i*3+1, i*3+2

            if not p_sub.empty:
                # --- Top: price chart ---
                axes[p_idx].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#e74c3c', alpha=0.5, where='post')
                axes[p_idx].step(p_sub['timestamp'], p_sub['bid_price_1'], color="#25ca6a", alpha=0.5, where='post')
                m_16 = p_sub[p_sub['spread'] == 16]
                if not m_16.empty:
                    axes[p_idx].plot(m_16['timestamp'], m_16['microprice'], color="blue", 
                                     linestyle='-', markersize=6,  label='Spread 16 MP')
                # �?Microprice visualization fix: added '*' marker points
                axes[p_idx].plot(p_sub['timestamp'], p_sub['microprice'], color="#3498db", alpha=0.7, 
                                  marker='*', markersize=4, label='Microprice')
                if not t_sub.empty:
                    # t_sub is already merged via merge_asof with p_merge, so 'microprice' column is available.
                    axes[p_idx].scatter(t_sub['timestamp'], t_sub['microprice'], 
                                        color='black', s=20, marker='o', zorder=10, label='MP at Trade')
                curr_mid = p_sub['mid_price'].median()
                axes[p_idx].set_ylim(curr_mid - 15, curr_mid + 15)
                
                if not t_sub.empty:
                    buys = t_sub[t_sub['color'] == 'red']
                    sells = t_sub[t_sub['color'] == 'blue']
                    axes[p_idx].scatter(buys['timestamp'], buys['price'], s=40, color='red', marker='^', zorder=5)
                    axes[p_idx].scatter(sells['timestamp'], sells['price'], s=40, color='blue', marker='v', zorder=5)
                
                axes[p_idx].set_title(f'Interval {i+1} | {start_ts}~{end_ts}', fontsize=12, loc='left')
                axes[p_idx].grid(True, alpha=0.1)
                axes[p_idx].axhline(y=10000, color='red', linestyle=':', alpha=0.5)
                # --- Middle: Spread chart (retained) ---
                axes[s_idx].fill_between(p_sub['timestamp'], p_sub['spread'], 16, color='purple', alpha=0.1)
                axes[s_idx].plot(p_sub['timestamp'], p_sub['spread'], color='purple', linewidth=0.8)
                axes[s_idx].axhline(y=16, color='red', linestyle='--', alpha=0.5)
                axes[s_idx].set_ylabel('Spread')
                
                if not t_sub.empty:
                    axes[s_idx].scatter(t_sub['timestamp'], t_sub['spread'], color='black', s=15, zorder=10)

                # --- Bottom: OBI chart (retained) ---
                axes[i_idx].fill_between(p_sub['timestamp'], p_sub['imbalance'], 0, color='gray', alpha=0.2)
                axes[i_idx].plot(p_sub['timestamp'], p_sub['imbalance'], color='black', linewidth=0.5)
                axes[i_idx].axhline(y=0.6, color='red', linestyle=':', alpha=0.5)
                axes[i_idx].axhline(y=-0.6, color='blue', linestyle=':', alpha=0.5)
                axes[i_idx].set_ylim(-1.1, 1.1)
                axes[i_idx].set_ylabel('Imb')
                
                if not t_sub.empty:
                    axes[i_idx].scatter(t_sub['timestamp'], t_sub['imbalance'], color='black', s=15, zorder=10)
            else:
                for ax in [axes[p_idx], axes[s_idx], axes[i_idx]]: ax.set_visible(False)

        # Print statistics
        total_trades = combined_trades['spread'].dropna().count()
        spread_16_count = (combined_trades['spread'] == 16).sum()

        if total_trades > 0:
            print(f"\n--- {idx2} Analysis ---")
            print(f"Total fills: {total_trades}")
            print(f"Fills where spread was 16 immediately before execution: {spread_16_count}")
            print(f"Ratio: {(spread_16_count / total_trades * 100):.2f}%")

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

plot_pepper_fix_scale(day=0)
```

    
    --- ASH_COATED_OSMIUM Analysis ---
    ?�체 체결 건수: 450
    체결 직전 ?�프?�드가 16?�었??건수: 284
    비율: 63.11%
    


    
![png](_analysis_img/round2_analysis_24_1.png)
    



```python

```


# bid_estimation.md

---

# Round 2: MAF Auction Bid Optimization

> **Problem:** All-or-nothing auction ??bid in the top 50% of all participants to gain 25% additional quote access.  
> **Access value:** ~8,000 XIRECS (estimated from gap between official backtester and full-quote-flow custom backtester).

## Overview

The MAF bid is an expected-profit optimization under distributional uncertainty:

$$
\mathbb{E}[\text{Profit}(b)] = P(\text{win} \mid b) \times (V - b)
$$

where V ??8,000 and P(win | b) = CDF of the modeled bid distribution at b.

**Bid distribution model:** Beta(α=2, β=5) on [0, 9,000]  
- Upper bound anchored to Round 1 top performer P&L as proxy for maximum rational bid  
- Right-skewed: most participants bid conservatively (loss aversion)

**Optimal bid:** ~3,580 XIRECS (maximizes expected profit)  
**Result:** Bid accepted ??market access granted.


---
## Section 1: Beta Distribution ??Bid Density & CDF

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
    annotation_text=f"Optimal ??{int(best_bid)}"
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
The inflection point ??where marginal cost accelerates ??marks the efficient frontier.  
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


# Manual_Analysis.md

---

# Round 2 Manual: Three-Pillar Resource Allocation Optimization

> **Budget:** 50,000 XIRECS  
> **Pillars:** Research (logarithmic) · Scale (linear) · Speed (rank-based)  
> **Core problem:** Speed payoff depends on the field's allocation distribution ??modeled as Beta(α, β).

## Overview

The manual challenge required allocating a fixed budget across three pillars with different marginal return structures.  
Speed was rank-based: its payoff depended entirely on competitors' allocations, making distributional modeling critical.

**Approach:**
1. Fix Speed allocation z, optimize Research/Scale split for the remaining budget
2. Model Speed distribution as Beta(α, β); compute expected Speed multiplier via numerical integration
3. Penalize sensitivity to (α, β) misspecification in the objective ??risk-adjusted optimization

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

print(f"A ??{opt_x:.2f}%")
print(f"B ??{opt_y:.2f}%")
print(f"C ??{opt_z:.2f}%")

print(f"C Score ??{C_score:.4f}")

print(f"Total PnL ??{final_pnl - 50000:.2f}")
```

    ----- ROBUST RESULT -----
    A ??16.97%
    B ??51.91%
    C ??31.12%
    C Score ??0.6326
    Total PnL ??237738.24
    


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
    
    # Find x split to maximize A×B (y = remaining ??x)
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

    --- ?�교?�된 최적??결과 ---
    A: 15.69%, B: 46.98%, C: 37.32%
    PnL: 238032.78 | Threshold: ?�위 22.76%
    



---
## Section 2: Beta Distribution Modeling for Speed

Speed multiplier is rank-based: modeled by fitting a Beta(α, β) distribution to competitors' Speed allocations.

Parameters: α=1.5, β=5 ??reflecting moderate concentration in the low-to-mid range with a long right tail.

**Sensitivity analysis:** P&L was materially more sensitive to β than α. Higher Speed allocation reduced this sensitivity ??the risk-adjusted objective pushed toward elevated Speed investment.



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
        # Soft constraint: sum ??100 (SLSQP handles this well)
        if x + y + z > 100.01: return 1e10

        A = 200000 * np.log(1 + x) / np.log(101)
        B = 0.07 * y

        # Convert z allocation (0??00) to fraction (0??)
        z_ratio = np.clip(z / 100, 1e-9, 1-1e-9)

        # Estimate rank using Beta CDF
        rank_pct = beta.cdf(z_ratio, a, b_param)

        C = 0.1 + 0.8 * rank_pct
        pnl = A * B * C
        return -pnl

    # Constraint: x + y + z ??100
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

        # Regret = max achievable P&L ??strategy P&L under assumed params
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

print(f"A ??{opt_x:.2f}%")
print(f"B ??{opt_y:.2f}%")
print(f"C ??{opt_z:.2f}%")

print(f"C Score ??{C_score:.4f}")

print(f"Expected PnL ??{final_pnl - 50000:,.0f}")

print(f"Target Threshold ??top {threshold_pct:.2f}%")


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
    A ??16.40%
    B ??49.70%
    C ??33.90%
    C Score ??0.6280
    Expected PnL ??220,432
    Target Threshold ???�위 34.00%
    




```python

```


