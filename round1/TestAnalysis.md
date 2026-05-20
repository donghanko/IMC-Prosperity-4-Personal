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

# 가격 Load data (mid_price 포함)
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# 거래 Load data (trade price 포함)
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. 자산별 시각화
assets = prices['product'].unique()

for asset in assets:
    # 해당 자산 데이터 필터링
    asset_prices = prices[prices['product'] == asset][['global_timestamp', 'mid_price']].copy()
    asset_trades = trades[trades['symbol'] == asset][['global_timestamp', 'price']].copy()
    asset_trades.rename(columns={'price': 'trade_price'}, inplace=True)
    
    # 두 데이터 병합 (Timestamp 기준)
    # 모든 가격 타임스탬프를 기준으로 거래 가격을 매칭
    combined = pd.merge(asset_prices, asset_trades, on='global_timestamp', how='left')
    
    # 중복 타임스탬프 처리 (같은 시간에 여러 거래가 있을 경우 마지막 거래가 채택됨)
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    # 없는 값은 이전 값으로 보간 (Forward Fill)
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill() # mid_price도 만약 비어있다면 보간
    
    # 3. Plotly를 이용한 상호작용 그래프 생성
    fig = go.Figure()

    # Mid Price 라인
    fig.add_trace(go.Scatter(
        x=combined['global_timestamp'],
        y=combined['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='rgba(100, 149, 237, 0.8)', width=1.5)
    ))

    # Trade Price 라인 (도트 또는 점선 스타일)
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
        template='plotly_dark', # 프리미엄한 느낌을 위해 다크 모드 적용
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

# 가격 Load data
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# 거래 Load data
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. 자산별 처리 및 시각화
assets = prices['product'].unique()

for asset in assets:
    asset_prices = prices[prices['product'] == asset][['global_timestamp', 'mid_price']].copy()
    asset_trades = trades[trades['symbol'] == asset][['global_timestamp', 'price']].copy()
    asset_trades.rename(columns={'price': 'trade_price'}, inplace=True)
    
    # [추가된 로직] 0값을 NaN으로 변경하여 보간 대상으로 만듦
    asset_prices['mid_price'] = asset_prices['mid_price'].replace(0, np.nan)
    asset_trades['trade_price'] = asset_trades['trade_price'].replace(0, np.nan)
    
    # 데이터 병합
    combined = pd.merge(asset_prices, asset_trades, on='global_timestamp', how='left')
    combined = combined.groupby('global_timestamp').last().reset_index()
    
    # [보간 처리] 이전 값으로 채우기 (Forward Fill)
    # 0이었던 값들과 매칭되지 않은 타임스탬프의 NaN 값들이 모두 이전 유효값으로 채워집니다.
    combined['trade_price'] = combined['trade_price'].ffill()
    combined['mid_price'] = combined['mid_price'].ffill()
    
    # 첫 부분에 값이 없어 남은 NaN은 다음 값으로 채움 (Optional: Bfill)
    combined['trade_price'] = combined['trade_price'].bfill()
    combined['mid_price'] = combined['mid_price'].bfill()
    
    # 3. 시각화
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

# 가격 Load data
prices_list = []
for f in price_files:
    df = pd.read_csv(f, sep=';')
    prices_list.append(df)
prices = pd.concat(prices_list)
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# 거래 Load data
trades_list = []
for f in trade_files:
    day = extract_day(f)
    df = pd.read_csv(f, sep=';')
    df['day'] = day
    trades_list.append(df)
trades = pd.concat(trades_list)
trades['global_timestamp'] = trades['day'] * 1000000 + trades['timestamp']

# 2. Fair Value 정의
def get_fair_value(asset, t):
    if 'PEPPER' in asset:
        return 10000 + 0.001 * t
    elif 'OSMIUM' in asset:
        return 10000
    else:
        return np.nan

# 3. 자산별 시각화 (Trade Deviation as Points)
assets = prices['product'].unique()

for asset in assets:
    # 해당 자산의 실제 거래 데이터 추출 및 0값 제거
    asset_trades = trades[trades['symbol'] == asset].copy()
    asset_trades = asset_trades[asset_trades['price'] > 0] # 0원인 사기 거래(?) 제거
    
    # [Fair Value 및 차이 계산] - 실제 거래 시점의 t 사용
    asset_trades['fair_value'] = asset_trades['global_timestamp'].apply(lambda t: get_fair_value(asset, t))
    asset_trades['deviation'] = asset_trades['price'] - asset_trades['fair_value']
    
    # 참고를 위한 Mid Price Deviation (선으로 연하게 표시)
    asset_prices = prices[prices['product'] == asset].copy()
    asset_prices = asset_prices[asset_prices['mid_price'] > 0]
    asset_prices['fair_value'] = asset_prices['global_timestamp'].apply(lambda t: get_fair_value(asset, t))
    asset_prices['deviation'] = asset_prices['mid_price'] - asset_prices['fair_value']

    # 4. 시각화
    fig = go.Figure()

    # Mid Price Deviation (흐릿한 선으로 배경 제공)
    fig.add_trace(go.Scatter(
        x=asset_prices['global_timestamp'],
        y=asset_prices['deviation'],
        mode='lines',
        name='Mid Price Deviation',
        line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
    ))

    # Trade Deviation (점으로 표시)
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

    # Zero Line (Fair Value 기준선)
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

# 연속적인 타임스탬프 계산
prices['global_timestamp'] = prices['day'] * 1000000 + prices['timestamp']

# 2. 오스뮴(ASH_COATED_OSMIUM) 데이터만 필터링
product = "ASH_COATED_OSMIUM"
osmium_data = prices[prices['product'] == product].copy()

# 0값(누락 데이터) 처리: 0을 NaN으로 바꾸고 이전 값으로 보간
for col in ['bid_price_1', 'ask_price_1', 'mid_price']:
    osmium_data[col] = osmium_data[col].replace(0, pd.NA).ffill().bfill()

# 3. 시각화
fig = go.Figure()

# Ask Price 1 (매도 호가 - 보통 위쪽에 위치)
fig.add_trace(go.Scatter(
    x=osmium_data['global_timestamp'],
    y=osmium_data['ask_price_1'],
    mode='lines',
    name='Ask Price 1',
    line=dict(color='rgba(255, 99, 71, 0.8)', width=1)
))

# Mid Price (중간 가격)
fig.add_trace(go.Scatter(
    x=osmium_data['global_timestamp'],
    y=osmium_data['mid_price'],
    mode='lines',
    name='Mid Price',
    line=dict(color='rgba(255, 255, 255, 0.9)', width=1.5)
))

# Bid Price 1 (매수 호가 - 보통 아래쪽에 위치)
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
    # 특정 구간을 자세히 보기 위한 슬라이더 추가
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

# 2. 오스뮴 데이터 필터링 및 편차 계산
product = "ASH_COATED_OSMIUM"
base_mid = 10000
osmium_trades = trades[trades['symbol'] == product].copy()
osmium_trades = osmium_trades[osmium_trades['price'] > 0] # 유효한 가격만

# 기준 가격(10000) 대비 편차 계산
osmium_trades['deviation'] = osmium_trades['price'] - base_mid

# 3. PDF (Probability Density Function) 시각화
# 히스토그램과 커널밀도추정(KDE)을 함께 그립니다.
hist_data = [osmium_trades['deviation'].tolist()]
group_labels = ['Trade Deviation (Price - 10000)']

fig = ff.create_distplot(
    hist_data, 
    group_labels, 
    bin_size=1, # 오스뮴은 가격 단위가 정수이므로 bin 크기를 1로 설정
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

# 0선(기준점) 표시
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

# 2. 오스뮴 데이터 필터링 및 매수/매도 구분
product = "ASH_COATED_OSMIUM"
base_mid = 10000
osmium_trades = trades[trades['symbol'] == product].copy()
osmium_trades = osmium_trades[osmium_trades['price'] > 0]

# 편차 계산
osmium_trades['deviation'] = osmium_trades['price'] - base_mid

# 매수(Buy): 체결가 > 10000 (보통 매도 호가 10002~10005 사이에서 체결)
# 매도(Sell): 체결가 < 10000 (보통 매수 호가 9995~9998 사이에서 체결)
buy_trades = osmium_trades[osmium_trades['deviation'] > 0]['deviation'].tolist()
sell_trades = osmium_trades[osmium_trades['deviation'] < 0]['deviation'].tolist()

# 3. PDF 시각화 (Buy: 빨간색, Sell: 파란색)
hist_data = [buy_trades, sell_trades]
group_labels = ['Taker Buy (Price > 10000)', 'Taker Sell (Price < 10000)']
colors = ['#FF4136', '#0074D9'] # Red for Buy, Blue for Sell

# 데이터가 없는 경우를 대비한 예외 처리
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

    # 10000 기준선 표시
    fig.add_shape(
        type="line", line=dict(color="white", width=2, dash="dash"),
        x0=0, x1=0, y0=0, y1=1,
        yref='paper'
    )

    fig.show()
else:
    print("분석할 충분한 매수/매도 거래 데이터가 없습니다.")

```




```python
import pandas as pd
import glob
import numpy as np
import plotly.figure_factory as ff
import re

# 1. Load data 및 전처리
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

# 2. 페퍼(PEPPER) 데이터 필터링 및 연속 타임스탬프 생성
product = "INTARIAN_PEPPER_ROOT"
# 전체 데이터의 시작일(보통 day -2)을 기준으로 연속적인 t(Global Timestamp) 계산
day_min = trades['day'].min() 
trades['global_ts'] = (trades['day'] - day_min) * 1000000 + trades['timestamp']

pepper_trades = trades[trades['symbol'] == product].copy()
pepper_trades = pepper_trades[pepper_trades['price'] > 0]

# Fair Value 및 편차 계산: Fair = 10000 + 0.001 * t
pepper_trades['fair_value'] = 10000 + 0.001 * pepper_trades['global_ts']
pepper_trades['deviation'] = pepper_trades['price'] - pepper_trades['fair_value']

# 매수(Buy): 체결가 > Fair (빨강), 매도(Sell): 체결가 < Fair (파랑)
buy_trades = pepper_trades[pepper_trades['deviation'] > 0]['deviation'].tolist()
sell_trades = pepper_trades[pepper_trades['deviation'] < 0]['deviation'].tolist()

# 3. PDF 시각화
if len(buy_trades) > 0 and len(sell_trades) > 0:
    hist_data = [buy_trades, sell_trades]
    group_labels = ['Pepper Taker Buy (Price > Fair)', 'Pepper Taker Sell (Price < Fair)']
    colors = ['#FF4136', '#0074D9'] 

    # 오타가 수정된 부분: create_distplot
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

    # 기준선(0) 표시
    fig.add_shape(
        type="line", line=dict(color="white", width=2, dash="dash"),
        x0=0, x1=0, y0=0, y1=1,
        yref='paper'
    )

    fig.show()
else:
    print("분석할 충분한 매수/매도 거래 데이터가 없습니다.")

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
    # Pepper Root만 필터링 및 시간순 정렬
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    pepper = pepper.sort_values(['day', 'timestamp']).reset_index(drop=True)
    return pepper

df = load_pepper_data()

# 2. '의도'와 '분위기' 계산
# Micro-price: 호가 잔량 가중치를 반영한 '진짜' 중간가 (의도 포착)
df['micro_price'] = (df['bid_price_1'] * df['ask_volume_1'] + 
                     df['ask_price_1'] * df['bid_volume_1']) / \
                    (df['bid_volume_1'] + df['ask_volume_1'])

# OBI (Order Book Imbalance): 매수세와 매도세의 비중 차이 (-1 ~ 1)
df['obi'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])

# '느린 흐름'을 보기 위한 지수 이동 평균 (Atmosphere)
df['obi_ema'] = df['obi'].ewm(span=100).mean()
df['intent_drift'] = (df['micro_price'] - df['mid_price']).rolling(50).mean()

# 3. 프리미엄 시각화 (서로 맞물리는 신호 확인)
plt.style.use('seaborn-v0_8-muted') # 깔끔한 테마 적용
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# 차트 1: 가격과 마이크로 프라이스 (의도가 가격을 선행하는지 확인)
ax1.plot(df['timestamp'], df['mid_price'], label='Mid Price', color='#bdc3c7', alpha=0.5)
ax1.plot(df['timestamp'], df['micro_price'], label='Micro Price (Intent)', color='#2980b9', linewidth=1)
ax1.set_title('Intarian Pepper Root: Price & Hidden Intent', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.2)

# 차트 2: 시장의 '분위기' (호가창이 어느 쪽으로 기울어 있는가)
ax2.fill_between(df['timestamp'], 0, df['obi_ema'], where=(df['obi_ema'] >= 0), color='#2ecc71', alpha=0.3, label='Buy Lean')
ax2.fill_between(df['timestamp'], 0, df['obi_ema'], where=(df['obi_ema'] < 0), color='#e74c3c', alpha=0.3, label='Sell Lean')
ax2.axhline(0, color='black', alpha=0.2, linestyle='--')
ax2.set_title('Market Atmosphere (Order Book Imbalance EMA)', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.2)

# 차트 3: 숨겨진 시그널 (의도와 실제 가격의 괴리 누적)
ax3.plot(df['timestamp'], df['intent_drift'], color='#8e44ad', label='Cumulative Signal (Micro - Mid)')
ax3.fill_between(df['timestamp'], 0, df['intent_drift'], color='#8e44ad', alpha=0.1)
ax3.axhline(0, color='black', alpha=0.2)
ax3.set_title('Hidden Signal: The Moment of Imbalance', fontsize=12)
ax3.set_xlabel('Timestamp')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# 힌트: 차트 3의 보라색 선이 0에서 멀어지는데 가격(차트 1)이 반응하지 않는다면, 
# 곧 그 방향으로 '툭' 하고 터질 가능성이 높습니다.

```


    
![png](TestAnalysis_files/TestAnalysis_10_0.png)
    



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_interpolated_pepper_days():
    # 1. 각 일차별 Load data (day -2, -1, 0)
    days = [-2, -1, 0]
    fig, axes = plt.subplots(len(days), 1, figsize=(15, 12), sharex=True)
    
    for i, day in enumerate(days):
        # 파일 읽기
        filename = f'prices_round_1_day_{day}.csv'
        try:
            df_day = pd.read_csv(filename, sep=';')
            pepper = df_day[df_day['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
            
            # 모든 타임스탬프(0~999,900)에 대해 비어있는 행 생성 및 보간
            all_timestamps = np.arange(0, 1000000, 100)
            full_index = pd.DataFrame({'timestamp': all_timestamps})
            pepper = pd.merge(full_index, pepper, on='timestamp', how='left')
            
            # 선형 보간 (Linear Interpolation) 적용
            pepper['mid_price'] = pepper['mid_price'].interpolate(method='linear')
            
            # 2. 데이터 시각화
            axes[i].plot(pepper['timestamp'], pepper['mid_price'], label=f'Day {day} Mid Price', color='#3498db', alpha=0.7)
            
            # 3. 추세 확인 (선형 회귀선 추가)
            # 페퍼의 '느린 성장' 성질을 시각적으로 확인하기 위함
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

# 실행
plot_interpolated_pepper_days()

```


    
![png](TestAnalysis_files/TestAnalysis_11_0.png)
    



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
            
            # 1. 가격이 0이거나 비어있는 데이터 모두 제거
            # zero-value 필터링 추가
            pepper = pepper[pepper['mid_price'] > 0]
            
            # 2. 시각화
            # 실선으로 보간하여 연결하고 실제 데이터는 점으로 표시
            axes[i].plot(pepper['timestamp'], pepper['mid_price'], 
                        linestyle='-', color='#3498db', alpha=0.6, label='Cleaned Path')
            
            axes[i].scatter(pepper['timestamp'], pepper['mid_price'], 
                           s=5, color='#2980b9', alpha=0.3, label='Actual Data')
            
            # 3. 우아한 추세선
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

# 실행
plot_pepper_cleaned()

```


    
![png](TestAnalysis_files/TestAnalysis_12_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_in_intervals(day=0, num_intervals=10):
    # 1. 특정 일차 Load data (기본값: Day 0)
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # Micro-price 계산 (상세 분석용)
        pepper['micro_price'] = (pepper['bid_price_1'] * pepper['ask_volume_1'] + 
                                pepper['ask_price_1'] * pepper['bid_volume_1']) / \
                                (pepper['bid_volume_1'] + pepper['ask_volume_1'])

        # 2. 구간 나누기
        interval_len = 1000000 // num_intervals
        
        # 10개의 서브플롯 생성
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            # 해당 구간 데이터 필터링
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # 중간가와 의도(Micro-price) 시각화
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

# 실행 (Day 0 데이터를 10구간으로 상세 분석)
plot_pepper_in_intervals(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_13_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_ultra_detailed(day=0, num_intervals=50):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # Micro-price 계산
        pepper['micro_price'] = (pepper['bid_price_1'] * pepper['ask_volume_1'] + 
                                pepper['ask_price_1'] * pepper['bid_volume_1']) / \
                                (pepper['bid_volume_1'] + pepper['ask_volume_1'])

        # 2. 구간 나누기 (20,000 타임스탬프 단위)
        interval_len = 1000000 // num_intervals
        
        # 50개의 서브플롯 생성
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                axes[i].plot(subset['timestamp'], subset['mid_price'], label='Mid', color='#7f8c8d', alpha=0.5)
                axes[i].plot(subset['timestamp'], subset['micro_price'], label='Micro', color='#e67e22', alpha=0.9)
                
                # 가독성을 위해 구간 범위만 제목으로 표시
                axes[i].set_title(f'[{i+1}] {start_ts} ~ {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
            else:
                axes[i].text(0.5, 0.5, 'Empty', ha='center', va='center', alpha=0.2)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Day 0를 50구간으로 쪼개서 정밀 분석 실행
plot_pepper_ultra_detailed(day=0, num_intervals=50)

```


    
![png](TestAnalysis_files/TestAnalysis_14_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_full_detail(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. 가격 및 체결 Load data
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        # 페퍼 루트 데이터만 추출
        p_pepper = prices[prices['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        t_pepper = trades[trades['symbol'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 2. 구간 나누기
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            # 구간 필터링
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = t_pepper[(t_pepper['timestamp'] >= start_ts) & (t_pepper['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # Best Bid / Ask (호가창 상태)
                # 'step'을 사용하면 호가가 변하는 모양을 더 정확히 볼 수 있습니다.
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], label='Ask', color='#e74c3c', alpha=0.4, where='post')
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], label='Bid', color='#2ecc71', alpha=0.4, where='post')
                
                # 심플 중간가 (흐름 참고용)
                axes[i].plot(p_sub['timestamp'], p_sub['mid_price'], color='#bdc3c7', linestyle=':', alpha=0.5)
                
                # Trade (실제 체결)
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

# 실행
plot_pepper_full_detail(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_15_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_trade_types(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load data 및 필터링 (0인 값 무시)
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        # 페퍼 추출 및 0값 제거
        p_pepper = prices[(prices['product'] == 'INTARIAN_PEPPER_ROOT') & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == 'INTARIAN_PEPPER_ROOT') & (trades['price'] > 0)].copy()
        
        # 2. 체결 타입 판별을 위한 조인 (Merge asof)
        # 각 체결 시점의 호가(Bid/Ask)를 알아야 매수/매도 판별이 가능합니다.
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')
        
        combined_trades = pd.merge_asof(t_pepper, 
                                        p_pepper[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')
        
        # 매수/매도 판별: Ask 근처면 Buy(빨강), Bid 근처면 Sell(파랑)
        def classify_trade(row):
            if row['price'] >= row['ask_price_1']: return 'red'  # Buy
            if row['price'] <= row['bid_price_1']: return 'blue' # Sell
            # 중간일 경우 더 가까운 쪽으로 판별
            if abs(row['price'] - row['ask_price_1']) < abs(row['price'] - row['bid_price_1']):
                return 'red'
            return 'blue'

        combined_trades['color'] = combined_trades.apply(classify_trade, axis=1)

        # 3. 구간 시각화
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 4 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # 호가 라인
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#e74c3c', alpha=0.2, where='post')
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], color='#2ecc71', alpha=0.2, where='post')
                
                # 체결 포인트 (매수/매도 구분)
                if not t_sub.empty:
                    # 매수 체결 (빨강)
                    buys = t_sub[t_sub['color'] == 'red']
                    axes[i].scatter(buys['timestamp'], buys['price'], s=30, color='red', marker='^', label='Buy Trade', zorder=5)
                    
                    # 매도 체결 (파랑)
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

# 실행
plot_pepper_trade_types(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_16_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_deep_book(day=0, num_intervals=50):
    price_file = f'prices_round_1_day_{day}.csv'
    trade_file = f'trades_round_1_day_{day}.csv'
    
    try:
        # 1. Load data 및 필터링
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        p_pepper = prices[(prices['product'] == 'INTARIAN_PEPPER_ROOT') & (prices['mid_price'] > 0)].copy()
        t_pepper = trades[(trades['symbol'] == 'INTARIAN_PEPPER_ROOT') & (trades['price'] > 0)].copy()
        
        # 체결 판별을 위한 조인
        p_pepper = p_pepper.sort_values('timestamp')
        t_pepper = t_pepper.sort_values('timestamp')
        combined_trades = pd.merge_asof(t_pepper, 
                                        p_pepper[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')

        combined_trades['color'] = combined_trades.apply(
            lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
            axis=1
        )

        # 2. 구간 시각화 (깊은 호가 포함)
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 5 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_sub = p_pepper[(p_pepper['timestamp'] >= start_ts) & (p_pepper['timestamp'] < end_ts)]
            t_sub = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_sub.empty:
                # --- 매도 호가 (Red 계열) ---
                # Level 3 (가장 깊은)
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_3'], color='#ff0000', alpha=0.1, where='post', label='Ask L3')
                # Level 2
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_2'], color='#ff0000', alpha=0.2, where='post', label='Ask L2')
                # Level 1 (가장 높은)
                axes[i].step(p_sub['timestamp'], p_sub['ask_price_1'], color='#ff0000', alpha=0.5, where='post', label='Ask L1', linewidth=1.5)
                
                # --- 매수 호가 (Green 계열) ---
                # Level 1 (가장 높은)
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_1'], color='#00aa00', alpha=0.5, where='post', label='Bid L1', linewidth=1.5)
                # Level 2
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_2'], color='#00aa00', alpha=0.2, where='post', label='Bid L2')
                # Level 3 (가장 깊은)
                axes[i].step(p_sub['timestamp'], p_sub['bid_price_3'], color='#00aa00', alpha=0.1, where='post', label='Bid L3')
                
                # 체결 포인트
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

# 실행
plot_pepper_deep_book(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_17_0.png)
    


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
        # 1. Load data 및 Filter target asset
        prices = pd.read_csv(price_file, sep=';')
        trades = pd.read_csv(trade_file, sep=';')
        
        p_sub_all = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
        t_sub_all = trades[(trades['symbol'] == product_name) & (trades['price'] > 0)].copy()
        
        # 체결 판별을 위한 조인
        p_sub_all = p_sub_all.sort_values('timestamp')
        t_sub_all = t_sub_all.sort_values('timestamp')
        combined_trades = pd.merge_asof(t_sub_all, 
                                        p_sub_all[['timestamp', 'bid_price_1', 'ask_price_1']], 
                                        on='timestamp')

        # 매수/매도 색상 판별
        combined_trades['color'] = combined_trades.apply(
            lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
            axis=1
        )

        # 2. 50구간 시각화
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 5 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            p_seg = p_sub_all[(p_sub_all['timestamp'] >= start_ts) & (p_sub_all['timestamp'] < end_ts)]
            t_seg = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
            
            if not p_seg.empty:
                # --- 매도 호가 (Red 계열) ---
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_3'], color='#ff0000', alpha=0.1, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_2'], color='#ff0000', alpha=0.2, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['ask_price_1'], color='#ff0000', alpha=0.5, where='post', linewidth=1.5, label='Ask L1')
                
                # --- 매수 호가 (Green 계열) ---
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_1'], color='#00aa00', alpha=0.5, where='post', linewidth=1.5, label='Bid L1')
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_2'], color='#00aa00', alpha=0.2, where='post')
                axes[i].step(p_seg['timestamp'], p_seg['bid_price_3'], color='#00aa00', alpha=0.1, where='post')
                
                # 체결 포인트 (삼각형)
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

# 오스뮴 분석 실행
plot_asset_full_analysis(product_name='ASH_COATED_OSMIUM', day=0)

# 만약 페퍼도 다시 보고 싶다면: 
# plot_asset_full_analysis(product_name='INTARIAN_PEPPER_ROOT', day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_19_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_spread_timeseries(day=0):
    price_file = f'prices_round_1_day_{day}.csv'
    
    try:
        # 1. Load data 및 스프레드 계산
        df = pd.read_csv(price_file, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 스프레드 계산: Ask L1 - Bid L1
        pepper['spread'] = pepper['ask_price_1'] - pepper['bid_price_1']
        
        # 2. 시각화
        plt.figure(figsize=(15, 6))
        
        # 원본 스프레드 (연한 색상)
        plt.plot(pepper['timestamp'], pepper['spread'], color='#3498db', alpha=0.3, label='Raw Spread')
        
        # 스프레드 이동 평균 (추세 확인용)
        pepper['spread_sma'] = pepper['spread'].rolling(window=100).mean()
        plt.plot(pepper['timestamp'], pepper['spread_sma'], color='#2c3e50', linewidth=2, label='100-tick Moving Average')
        
        plt.title(f'Intarian Pepper Root: Bid-Ask Spread Time Series (Day {day})', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp')
        plt.ylabel('Spread Amount')
        
        # 통계 정보 추가
        mean_spread = pepper['spread'].mean()
        plt.axhline(mean_spread, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_spread:.2f}')
        
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
        
        # 간단한 통계 출력
        print(f"--- Spread Statistics for Day {day} ---")
        print(f"Average Spread: {mean_spread:.2f}")
        print(f"Max Spread: {pepper['spread'].max()}")
        print(f"Min Spread: {pepper['spread'].min()}")
        print(f"Standard Deviation: {pepper['spread'].std():.2f}")

    except Exception as e:
        print(f"Error plotting spread: {e}")

# 실행
plot_pepper_spread_timeseries(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_20_0.png)
    


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
            
            # 결측치 및 0값 제외
            pepper = pepper[pepper['spread'] > 0]
            
            # 1. 시각화 (원본 + 추세선)
            axes[i].plot(pepper['timestamp'], pepper['spread'], color='#3498db', alpha=0.2)
            
            # 이동 평균 (전체적인 흐름)
            sma = pepper['spread'].rolling(window=500).mean()
            axes[i].plot(pepper['timestamp'], sma, color='#e74c3c', linewidth=2, label='Trend')
            
            # 선형 회귀로 하루 안에서의 증가/감소 경향성 파악
            z = np.polyfit(pepper['timestamp'], pepper['spread'], 1)
            p = np.poly1d(z)
            axes[i].plot(pepper['timestamp'], p(pepper['timestamp']), "k--", alpha=0.8, label='Linear fit')
            
            axes[i].set_title(f'Day {day} Spread', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Timestamp')
            if i == 0: axes[i].set_ylabel('Spread Amount')
            axes[i].grid(True, alpha=0.1)
            axes[i].legend(fontsize=8)
            
            # 통계 저장
            daily_stats.append({
                'day': day,
                'mean': pepper['spread'].mean(),
                'slope': z[0] * 1000000 # 100만 타임스탬프당 변화량
            })
            
        except FileNotFoundError:
            print(f"File {filename} not found.")

    plt.suptitle('Intarian Pepper Root: Spread Trend Analysis (Over 3 Days)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 통계 출력 (일자별 비교)
    print("\n--- Inter-Day Spread Trend Statistics ---")
    stats_df = pd.DataFrame(daily_stats)
    print(stats_df.to_string(index=False))
    
    if stats_df['mean'].is_monotonic_increasing:
        print("\n[!] Finding: 일차별 평균 스프레드가 꾸준히 '증가'하고 있습니다.")
    elif stats_df['mean'].is_monotonic_decreasing:
        print("\n[!] Finding: 일차별 평균 스프레드가 꾸준히 '감소'하고 있습니다.")
    else:
        print("\n[!] Finding: 일차별로 특별한 선형 증가/감소 패턴은 보이지 않습니다.")

# 실행
analyze_pepper_spread_trend()

```


    
![png](TestAnalysis_files/TestAnalysis_21_0.png)
    


    
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
            
            # 결측치 및 0값 제외
            asset_df = asset_df[asset_df['spread'] > 0]
            
            # 1. 시각화
            axes[i].plot(asset_df['timestamp'], asset_df['spread'], color='#16a085', alpha=0.2)
            
            # 이동 평균
            sma = asset_df['spread'].rolling(window=500).mean()
            axes[i].plot(asset_df['timestamp'], sma, color='#d35400', linewidth=2, label='Trend')
            
            # 선형 회귀
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
    
    # 통계 출력
    stats_df = pd.DataFrame(daily_stats)
    print(f"\n--- {product_name} Inter-Day Spread Statistics ---")
    print(stats_df.to_string(index=False))

# 오스뮴 분석 실행
analyze_asset_spread_trend('ASH_COATED_OSMIUM')

```


    
![png](TestAnalysis_files/TestAnalysis_22_0.png)
    


    
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
        
        # 1. 0.0값을 NaN으로 바꾸고 이전 값으로 채우기 (ffill)
        # 먼저 0을 제거해야 '이전 정상값'이 유지됩니다.
        pepper.loc[pepper['mid_price'] <= 0, 'mid_price'] = np.nan
        pepper['mid_price'] = pepper['mid_price'].ffill()
        
        # 2. 모든 타임스탬프(0~999,900)에 대해 빈 구간 채우기 (사용자 요청 bfill)
        all_ts = np.arange(0, 1000000, 100)
        full_df = pd.DataFrame({'timestamp': all_ts})
        pepper = pd.merge(full_df, pepper, on='timestamp', how='left')
        
        # 빈 타임스탬프 구간은 bfill 적용
        pepper['mid_price'] = pepper['mid_price'].bfill()
        
        # 3. 점프 분석 (직전 값과의 차이)
        pepper['price_diff'] = pepper['mid_price'].diff().abs()
        
        # Finding 필터링 (0보다 큰 변동만 확인)
        jumps = pepper[pepper['price_diff'] > 0].copy()
        
        print(f"--- Pepper Price Jump Analysis (Day {day}, Cleaned Zeroes) ---")
        if not jumps.empty:
            print(f"최대 점프 폭: {jumps['price_diff'].max()}")
            print(f"최소 점프 폭: {jumps['price_diff'].min()}")
            
            # 상위 10개 큰 점프 확인
            top_10 = jumps.sort_values('price_diff', ascending=False).head(10)
            print("\n[상위 10개 점프 지점]")
            print(top_10[['timestamp', 'price_diff', 'mid_price']].to_string(index=False))
        else:
            print("분석할 점프 데이터가 없습니다.")
            
    except Exception as e:
        print(f"Error: {e}")

# Day 0를 기준으로 0.0 필터링 후 점프 분석 실행
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
        # 0인 값 제외하고 페퍼 데이터만 추출
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        # 구간 나누기 (약 33,333 타임스탬프 단위)
        interval_len = 1000000 // num_intervals
        
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # 가격 흐름 시각화
                axes[i].plot(subset['timestamp'], subset['mid_price'], color='#2c3e50', linewidth=1.5)
                
                # 가독성을 위한 설정
                axes[i].set_title(f'Interval {i+1}: {start_ts} to {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                
                # Y축 범위를 데이터에 맞춰 타이트하게 설정 (상세 변화 확인용)
                y_min, y_max = subset['mid_price'].min(), subset['mid_price'].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data (Zeroes Filtered)', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# Day 0의 미드 가격을 30구간으로 상세 분석
plot_pepper_mid_30_intervals(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_26_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_day_0_mid():
    # Day 0 file (competition day 3)
    filename = 'prices_round_1_day_0.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # 노이즈 제거 및 페퍼 데이터 필터링
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        plt.figure(figsize=(15, 7))
        plt.plot(pepper['timestamp'], pepper['mid_price'], color='#2c3e50', linewidth=1, label='Pepper L3 Mid Price')
        
        # 추세선 추가
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
        
        print(f"L3 시작 가격: {pepper['mid_price'].iloc[0]}")
        print(f"L3 종료 가격: {pepper['mid_price'].iloc[-1]}")
        print(f"L3 평균 가격: {pepper['mid_price'].mean():.2f}")

    except FileNotFoundError:
        print(f"File {filename} not found. 데이터 파일 이름을 확인해주세요.")

# 실행
plot_day_0_mid()

```


    
![png](TestAnalysis_files/TestAnalysis_27_0.png)
    


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
        # 0 제외 및 페퍼 필터링
        pepper = df[(df['product'] == 'INTARIAN_PEPPER_ROOT') & (df['mid_price'] > 0)].copy()
        
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # 가격 시각화
                axes[i].plot(subset['timestamp'], subset['mid_price'], color='#34495e', linewidth=1.5)
                
                # 구간 정보 및 Y축 자동 조정
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

# L3 데이터를 30구간으로 나누어 분석 실행
plot_l3_mid_30_intervals(num_intervals=30)

```


    
![png](TestAnalysis_files/TestAnalysis_28_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l3_30_intervals(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Level 3 Mid Price 계산 (비드 3 + 애스크 3) / 2
        # 데이터가 0인 구간은 제외
        pepper = pepper[(pepper['bid_price_3'] > 0) & (pepper['ask_price_3'] > 0)]
        pepper['mid_l3'] = (pepper['bid_price_3'] + pepper['ask_price_3']) / 2.0
        
        # 2. 30구간 나누기
        interval_len = 1000000 // num_intervals
        fig, axes = plt.subplots(num_intervals, 1, figsize=(15, 3 * num_intervals))
        
        for i in range(num_intervals):
            start_ts = i * interval_len
            end_ts = (i + 1) * interval_len
            
            subset = pepper[(pepper['timestamp'] >= start_ts) & (pepper['timestamp'] < end_ts)]
            
            if not subset.empty:
                # Level 3 Mid 시각화
                axes[i].plot(subset['timestamp'], subset['mid_l3'], color='#8e44ad', linewidth=1.5, label='L3 Mid')
                
                # 최상단 Mid(L1)와 비교하고 싶다면 아래 주석 해제
                # axes[i].plot(subset['timestamp'], subset['mid_price'], color='#bdc3c7', alpha=0.5, label='L1 Mid')
                
                axes[i].set_title(f'Level 3 Mid Interval {i+1}: {start_ts} to {end_ts}', fontsize=10, loc='left')
                axes[i].grid(True, alpha=0.1)
                axes[i].legend(loc='upper right', fontsize=8)
                
                # Y축 범위 조정
                y_min, y_max = subset['mid_l3'].min(), subset['mid_l3'].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No L3 Data', ha='center', va='center', alpha=0.2)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File {filename} not found.")

# 페퍼의 Level 3 기반 중간가 분석 (30구간) 실행
plot_pepper_mid_l3_30_intervals(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_29_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pepper_mid_l3_robust(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # 페퍼 추출
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Level 3 가격 계산 (데이터형 변환 및 숫자만 추출)
        # 0이나 비어있는 값은 제외하되, '데이터가 있는 곳만' 남깁니다.
        pepper['bid_price_3'] = pd.to_numeric(pepper['bid_price_3'], errors='coerce')
        pepper['ask_price_3'] = pd.to_numeric(pepper['ask_price_3'], errors='coerce')
        
        # L3 데이터가 하나라도 있는 구간을 찾기 위해 필터링
        pepper_l3 = pepper.dropna(subset=['bid_price_3', 'ask_price_3']).copy()
        pepper_l3 = pepper_l3[(pepper_l3['bid_price_3'] > 0) & (pepper_l3['ask_price_3'] > 0)]
        
        if pepper_l3.empty:
            print(f"⚠️ 경고: {filename} 파일에 Pepper의 Level 3 호가 데이터가 하나도 없습니다!")
            # 레벨 1이라도 보여줌
            print("대신 Level 1 중간가(Mid Price)를 확인합니다.")
            pepper_l3 = pepper[(pepper['mid_price'] > 0)].copy()
            target_col = 'mid_price'
        else:
            pepper_l3['mid_l3'] = (pepper_l3['bid_price_3'] + pepper_l3['ask_price_3']) / 2.0
            target_col = 'mid_l3'

        # 2. 30구간 나누기
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
                
                # Y축 범위 자동 최적화
                y_min, y_max = subset[target_col].min(), subset[target_col].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 2.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data in this range', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"오류 발생: {e}")

# 실행
plot_pepper_mid_l3_robust(day=0)

```

    ⚠️ 경고: prices_round_1_day_0.csv 파일에 Pepper의 Level 3 호가 데이터가 하나도 없습니다!
    대신 Level 1 중간가(Mid Price)를 확인합니다.
    


    
![png](TestAnalysis_files/TestAnalysis_30_1.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l2_30_intervals(day=0, num_intervals=30):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        # 페퍼 추출
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # 1. Level 2 가격 수치 변환 (NaN 및 0 처리)
        pepper['bid_price_2'] = pd.to_numeric(pepper['bid_price_2'], errors='coerce')
        pepper['ask_price_2'] = pd.to_numeric(pepper['ask_price_2'], errors='coerce')
        
        # L2 데이터가 존재하는 지점만 추출
        pepper_l2 = pepper.dropna(subset=['bid_price_2', 'ask_price_2']).copy()
        pepper_l2 = pepper_l2[(pepper_l2['bid_price_2'] > 0) & (pepper_l2['ask_price_2'] > 0)]
        
        if pepper_l2.empty:
            print(f"⚠️ {filename}에 Level 2 데이터도 충분하지 않습니다. 자동으로 Level 1(Mid)을 사용합니다.")
            pepper_l2 = pepper[pepper['mid_price'] > 0].copy()
            pepper_l2['target_price'] = pepper_l2['mid_price']
            tag = "Level 1 Mid"
        else:
            pepper_l2['target_price'] = (pepper_l2['bid_price_2'] + pepper_l2['ask_price_2']) / 2.0
            tag = "Level 2 Mid"

        # 2. 30구간 나누기 시각화
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
                
                # Y축 자동 스케일링
                y_min, y_max = subset['target_price'].min(), subset['target_price'].max()
                margin = (y_max - y_min) * 0.2 if y_max != y_min else 2.0
                axes[i].set_ylim(y_min - margin, y_max + margin)
            else:
                axes[i].text(0.5, 0.5, 'No Data (Interval Empty)', ha='center', va='center', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"오류: {e}")

# Level 2 중간가 기반 30구간 분석 실행
plot_pepper_mid_l2_30_intervals(day=0)

```


    
![png](TestAnalysis_files/TestAnalysis_31_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_pepper_mid_l2_ultra_detailed(day=0, num_intervals=100):
    filename = f'prices_round_1_day_{day}.csv'
    try:
        df = pd.read_csv(filename, sep=';')
        pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
        
        # Level 2 가격 Quantification
        pepper['bid_price_2'] = pd.to_numeric(pepper['bid_price_2'], errors='coerce')
        pepper['ask_price_2'] = pd.to_numeric(pepper['ask_price_2'], errors='coerce')
        
        # 유효 데이터 추출 (L2가 있는 곳 위주)
        pepper_l2 = pepper.dropna(subset=['bid_price_2', 'ask_price_2']).copy()
        pepper_l2 = pepper_l2[(pepper_l2['bid_price_2'] > 0) & (pepper_l2['ask_price_2'] > 0)]
        
        if pepper_l2.empty:
            pepper_l2 = pepper[pepper['mid_price'] > 0].copy()
            pepper_l2['target_price'] = pepper_l2['mid_price']
            tag = "L1 Mid"
        else:
            pepper_l2['target_price'] = (pepper_l2['bid_price_2'] + pepper_l2['ask_price_2']) / 2.0
            tag = "L2 Mid"

        # 100개 구간 시각화 (더 컴팩트한 레이아웃)
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
        print(f"오류: {e}")

# Level 2 기준 100구간 울트라 디테일 분석
plot_pepper_mid_l2_ultra_detailed(day=0, num_intervals=100)

```


    
![png](TestAnalysis_files/TestAnalysis_32_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_asset_3day_detailed(product_name='ASH_COATED_OSMIUM', num_intervals_per_day=10):
    days = [-2, -1, 0]
    total_intervals = len(days) * num_intervals_per_day
    
    # 30개(3일 * 10구간)의 서브플롯 생성
    fig, axes = plt.subplots(total_intervals, 1, figsize=(15, 5 * total_intervals))
    
    interval_idx = 0
    for day in days:
        price_file = f'prices_round_1_day_{day}.csv'
        trade_file = f'trades_round_1_day_{day}.csv'
        
        try:
            # 1. Load data 및 필터링
            prices = pd.read_csv(price_file, sep=';')
            trades = pd.read_csv(trade_file, sep=';')
            
            p_asset = prices[(prices['product'] == product_name) & (prices['mid_price'] > 0)].copy()
            t_asset = trades[(trades['symbol'] == product_name) & (trades['price'] > 0)].copy()
            
            # 매수/매도 체결 판별을 위한 조인
            p_asset = p_asset.sort_values('timestamp')
            t_asset = t_asset.sort_values('timestamp')
            combined_trades = pd.merge_asof(t_asset, p_asset[['timestamp', 'bid_price_1', 'ask_price_1']], on='timestamp')
            
            combined_trades['color'] = combined_trades.apply(
                lambda r: 'red' if r['price'] >= r['ask_price_1'] else ('blue' if r['price'] <= r['bid_price_1'] else 'gray'), 
                axis=1
            )

            # 2. 하루를 10구간으로 나누어 시각화
            interval_len = 1000000 // num_intervals_per_day
            for i in range(num_intervals_per_day):
                start_ts = i * interval_len
                end_ts = (i + 1) * interval_len
                
                p_seg = p_asset[(p_asset['timestamp'] >= start_ts) & (p_asset['timestamp'] < end_ts)]
                t_seg = combined_trades[(combined_trades['timestamp'] >= start_ts) & (combined_trades['timestamp'] < end_ts)]
                
                ax = axes[interval_idx]
                if not p_seg.empty:
                    # 매도 호가 (L1~L3)
                    ax.step(p_seg['timestamp'], p_seg['ask_price_3'], color='red', alpha=0.1, where='post')
                    ax.step(p_seg['timestamp'], p_seg['ask_price_2'], color='red', alpha=0.2, where='post')
                    ax.step(p_seg['timestamp'], p_seg['ask_price_1'], color='red', alpha=0.5, where='post', linewidth=1.2, label='Ask L1')
                    
                    # 매수 호가 (L1~L3)
                    ax.step(p_seg['timestamp'], p_seg['bid_price_1'], color='green', alpha=0.5, where='post', linewidth=1.2, label='Bid L1')
                    ax.step(p_seg['timestamp'], p_seg['bid_price_2'], color='green', alpha=0.2, where='post')
                    ax.step(p_seg['timestamp'], p_seg['bid_price_3'], color='green', alpha=0.1, where='post')
                    
                    # 체결 포인트
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

# 실행 (오스뮴 3일치 분석)
plot_asset_3day_detailed(product_name='ASH_COATED_OSMIUM', num_intervals_per_day=10)

# 페퍼를 분석하고 싶다면 아래 주석 해제
# plot_asset_3day_detailed(product_name='INTARIAN_PEPPER_ROOT', num_intervals_per_day=10)

```


    
![png](TestAnalysis_files/TestAnalysis_33_0.png)
    

