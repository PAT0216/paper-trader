# Detailed Implementation Guide: Two High-Performance ML Trading Strategies

## Part 1: Dual-Task MLP on Behavioral Alpha Factors (2024)

### Overview
- **Architecture:** Multilayer Perceptron with dual output heads
- **Input:** 40 engineered behavioral/technical alpha factors (OHLCV-derived)
- **Output Heads:** 
  1. Regression head: predicts 5-day forward return (continuous)
  2. Classification head: predicts up/down direction (binary)
- **Performance:** IC=0.0340, IR=1.6075, Sharpe=1.61, Top-5 long/short 800%+ (5-year)

---

## 1.1 The 40 Behavioral Alpha Factors (Complete List)

These factors are derived from open, high, low, close, volume (OHLCV) and standard technical indicators (MACD, RSI, Bollinger Bands). They are grouped into 3 behavioral themes:

### **Group 1: Momentum & Herding (Positive Contribution)**
| # | Factor Name | Formula | Logic |
|---|---|---|---|
| 1 | `alpha_signal_open_down` | `(Open < Open.shift(1)).astype(int)` | Binary: 1 if open dropped vs prior day |
| 2 | `alpha_kline_body_strength` | `(Close - Open) / (High - Low + 0.001)` | Candle body size as % of range (0–1); bullish if close > open |
| 3 | `alpha_macd_rsi_product` | `macd_diff * rsi_14` | MACD (histogram) × RSI(14); momentum × oscillator |
| 4 | `alpha_rsi_times_vwapdev_rank` | `rsi_14 * rank(Close - vwap)` | RSI × deviation from VWAP; capture overbought with price divergence |
| 5 | `alpha_volspike_times_body` | `(Volume / ma5_volume) * alpha_kline_body_strength` | Volume surge × candle strength; herd buying |
| 6 | `alpha_ma200diff_times_rsi` | `rank(ma200 - Close) * rsi_14` | RSI weighted by distance from 200-day MA |
| 7 | `alpha_shortdiff_times_macd` | `rank(Close - ma5) * rank(Close - ma20) * macd_diff` | Short MA crossover × MACD; momentum confirmation |
| 8 | `alpha_median_dev_times_rsi` | `((Close - (High+Low)/2) / (High-Low+0.001)) * rsi_14` | Median price deviation × RSI |
| 9 | `alpha_macd_times_lowdev` | `macd_diff * (Close - Low) / (High - Low + 0.001)` | MACD × proximity to daily low; buy signal quality |
| 10 | `alpha_macd_times_volatility` | `macd_diff * std(Close, 10)` | MACD × 10-day volatility; momentum in volatile regime |
| 11 | `alpha_rsi_bounce_strength` | Indicator(rsi_14 rising & <30) * (rsi_14 - 50) | RSI oversold bounce strength |
| 12 | `alpha_body_times_rsi` | `alpha_kline_body_strength * rsi_14` | Strong candle × RSI level |

### **Group 2: Volume-Price Divergence (Mixed/Reversal Signals)**
| # | Factor Name | Formula | Logic |
|---|---|---|---|
| 13 | `alpha_volume_spike_ratio` | `Volume / ma(Volume, 5)` | Today's volume vs 5-day MA; spike indicates conviction |
| 14 | `alpha_close_vwap_diff_rank` | `rank(Close - vwap)` | Price vs Volume-Weighted Average Price; deviation from fair value |
| 15 | `alpha_close_near_low` | `(High - Close) / (High - Low + 0.001)` | How close to daily low; reversal readiness |
| 16 | `alpha_short_ma_diff_rank_mul` | `rank(Close - ma5) * rank(Close - ma20)` | Short MA position (5 × 20); crossover indicator |
| 17 | `alpha_volume_times_macd` | `Volume * macd_diff` | Volume × MACD; volume-confirmed momentum |
| 18 | `alpha_macd_per_volume` | `macd_diff / Volume` | MACD relative to volume; signal quality |
| 19 | `alpha_close_range_high_rank` | `rank((Close - Low) / (High - Low + 0.001))` | Price position in daily range |
| 20 | `alpha_volatility_times_rsi` | `std(Close, 10) * rsi_14` | Volatility × RSI; risk-adjusted signal |
| 21 | `alpha_vwapdev_over_macd` | `rank(Close - vwap) / (macd_diff + 1e-6)` | VWAP divergence normalized by MACD |

### **Group 3: Oversold Reversals (Bottom Detection)**
| # | Factor Name | Formula | Logic |
|---|---|---|---|
| 22 | `alpha_close_delta_1d` | `-1 * Close.diff()` | Negative return; -1× is reversal signal |
| 23 | `alpha_momentum_5d_min_rank` | `-1 * rank(Close - Close.shift(5))` | Negative 5-day momentum (if negative, oversold) |
| 24 | `alpha_ma200_close_diff_rank` | `rank(ma200 - Close)` | Ranking: when MA200 > Close, stock far from LT MA (potential bounce) |
| 25 | `alpha_volatility_10d_rank_neg` | `-1 * rank(std(Close, 10))` | Negative vol rank; low vol after decline = reversal |
| 26 | `alpha_rev_rsi_times_lowdev` | `(100 - rsi_14) * (Close - Low) / (High - Low + 0.001)` | (1 - RSI) × proximity to low; double confirmation of oversold |
| 27 | `alpha_volrank_times_revrsi` | `-1 * rank(std(Close, 10)) * (100 - rsi_14)` | Low vol × low RSI = washout |
| 28 | `alpha_rsi_vs_50` | `rsi_14 - 50` | RSI distance from neutral (50); <0 = oversold |
| 29 | `alpha_rsi_bounce_rank` | `rank((rsi_14 <30 & rsi_14↑) * (rsi_14 - 50))` | Ranked bounce strength |
| 30 | `alpha_macd_cross` | `Indicator(macd_diff > 0 & macd_diff.shift(1) ≤ 0)` | MACD golden cross (0 line above); bullish reversal |
| 31 | `alpha_macd_cross_strength` | `alpha_macd_cross * macd_diff` | MACD cross × magnitude |
| 32 | `alpha_price_vs_boll` | `Indicator(Close < boll_lower) * Close` | Price below lower Bollinger Band; extreme oversold |
| 33 | `alpha_boll_rebound` | `Indicator(Close_yesterday < boll_lower & Close_today > boll_lower)` | Bounce off BB lower band |

### **Group 4: Advanced Trend & Oscillator Combinations**
| # | Factor Name | Formula | Logic |
|---|---|---|---|
| 34 | `alpha_close_median_dev_ratio` | `(Close - (High+Low)/2) / (High - Low + 0.001)` | Median deviation; -1 to 1 scale |
| 35 | `alpha_macd_rank_times_rsi` | `rank(macd_diff) * rsi_14` | Ranked MACD × raw RSI |
| 36 | `alpha_macd_rank_times_price` | `rank(macd_diff) * Close` | Ranked MACD × price level |
| 37 | `alpha_rsi_rank_times_close` | `rank(rsi_14) * Close` | Ranked RSI × price |
| 38 | `alpha_rsi_rise_velocity` | `rsi_14 - rsi_14.shift(3)` | 3-day RSI change; momentum of oscillator |
| 39 | `alpha_boll_bandwidth` | `boll_upper - boll_lower` | Bollinger Band width; volatility measure |
| 40 | `alpha_close_boll_mid_ratio` | `(Close - boll_mid) / (boll_upper - boll_lower + 1e-6)` | Price position within BB range (-1 to 1) |

**Note:** VWAP = Volume-Weighted Average Price; MACD_diff = MACD line − Signal line; RSI(14) = 14-period Relative Strength Index; Bollinger Bands = SMA(20) ± 2×std(20)

---

## 1.2 Architecture & Training

### **Network Architecture**

```python
import torch
import torch.nn as nn

class DualTaskMLP(nn.Module):
    def __init__(self, input_dim=40, hidden1=64, hidden2=32, dropout=0.1):
        super(DualTaskMLP, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Regression head: predict 5-day forward return (continuous)
        self.reg_head = nn.Linear(hidden2, 1)
        
        # Classification head: predict up/down (binary)
        self.clf_head = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Shared feature extraction
        h1 = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout2(torch.relu(self.bn2(self.fc2(h1))))
        
        # Regression output (no activation; will clip to [-0.3, 0.3] for returns)
        reg_out = self.reg_head(h2)
        
        # Classification output (sigmoid for binary: 0=down, 1=up)
        clf_out = self.sigmoid(self.clf_head(h2))
        
        return reg_out, clf_out
```

### **Training Configuration**

```python
# Hyperparameters
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 100
BATCH_SIZE = 2048
GRADIENT_CLIP = 0.5
PATIENCE = 15  # Early stopping

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Loss function: weighted combination
# L_total = L_reg + 0.5 * L_clf
criterion_reg = nn.MSELoss()
criterion_clf = nn.BCELoss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.7, 
    patience=5, 
    verbose=True
)
```

### **Training Loop**

```python
def train_epoch(model, train_loader, optimizer, criterion_reg, criterion_clf, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_reg_batch, y_clf_batch in train_loader:
        X_batch = X_batch.to(device)
        y_reg_batch = y_reg_batch.to(device)
        y_clf_batch = y_clf_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reg_out, clf_out = model(X_batch)
        
        # Loss calculation
        loss_reg = criterion_reg(reg_out.squeeze(), y_reg_batch)
        loss_clf = criterion_clf(clf_out.squeeze(), y_clf_batch)
        loss_total = loss_reg + 0.5 * loss_clf
        
        # Backward pass with gradient clipping
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss_total.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion_reg, criterion_clf, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_reg_batch, y_clf_batch in val_loader:
            X_batch = X_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            y_clf_batch = y_clf_batch.to(device)
            
            reg_out, clf_out = model(X_batch)
            loss_reg = criterion_reg(reg_out.squeeze(), y_reg_batch)
            loss_clf = criterion_clf(clf_out.squeeze(), y_clf_batch)
            loss_total = loss_reg + 0.5 * loss_clf
            
            total_loss += loss_total.item()
    
    return total_loss / len(val_loader)
```

---

## 1.3 Feature Engineering & Preprocessing

### **Data Preparation**

```python
import pandas as pd
import numpy as np
from talib import MACD, RSI, BBANDS

def compute_alpha_factors(df):
    """
    Input: df with columns [Open, High, Low, Close, Volume] per stock-day
    Output: df with 40 alpha factors
    """
    
    # Basic MA/EMA
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma200'] = df['Close'].rolling(200).mean()
    
    # VWAP (simplified: cumulative sum of price*volume / cumulative volume)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap'] = (df['typical_price'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Technical indicators
    df['macd_diff'], _, df['macd_signal'] = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['rsi_14'] = RSI(df['Close'], timeperiod=14)
    
    boll_upper, boll_mid, boll_lower = BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['boll_upper'] = boll_upper
    df['boll_mid'] = boll_mid
    df['boll_lower'] = boll_lower
    
    # Initialize factors dataframe
    factors = pd.DataFrame(index=df.index)
    
    # Factor 1-12: Momentum & Herding
    factors['alpha_signal_open_down'] = (df['Open'] < df['Open'].shift(1)).astype(int)
    factors['alpha_kline_body_strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.001)
    factors['alpha_macd_rsi_product'] = df['macd_diff'] * df['rsi_14']
    factors['alpha_rsi_times_vwapdev_rank'] = df['rsi_14'] * df['Close'].sub(df['vwap']).abs().rank(pct=True)
    factors['alpha_volspike_times_body'] = (df['Volume'] / df['Volume'].rolling(5).mean()) * factors['alpha_kline_body_strength']
    factors['alpha_ma200diff_times_rsi'] = (df['ma200'] - df['Close']).rank(pct=True) * df['rsi_14']
    factors['alpha_shortdiff_times_macd'] = ((df['Close'] - df['ma5']).rank(pct=True) * (df['Close'] - df['ma20']).rank(pct=True)) * df['macd_diff']
    factors['alpha_median_dev_times_rsi'] = ((df['Close'] - (df['High'] + df['Low']) / 2) / (df['High'] - df['Low'] + 0.001)) * df['rsi_14']
    factors['alpha_macd_times_lowdev'] = df['macd_diff'] * ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001))
    factors['alpha_macd_times_volatility'] = df['macd_diff'] * df['Close'].rolling(10).std()
    factors['alpha_rsi_bounce_strength'] = ((df['rsi_14'] < 30) & (df['rsi_14'] > df['rsi_14'].shift(1))).astype(int) * (df['rsi_14'] - 50)
    factors['alpha_body_times_rsi'] = factors['alpha_kline_body_strength'] * df['rsi_14']
    
    # Factor 13-21: Volume-Price Divergence
    factors['alpha_volume_spike_ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()
    factors['alpha_close_vwap_diff_rank'] = (df['Close'] - df['vwap']).rank(pct=True)
    factors['alpha_close_near_low'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 0.001)
    factors['alpha_short_ma_diff_rank_mul'] = (df['Close'] - df['ma5']).rank(pct=True) * (df['Close'] - df['ma20']).rank(pct=True)
    factors['alpha_volume_times_macd'] = df['Volume'] * df['macd_diff']
    factors['alpha_macd_per_volume'] = df['macd_diff'] / (df['Volume'] + 1e-6)
    factors['alpha_close_range_high_rank'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)).rank(pct=True)
    factors['alpha_volatility_times_rsi'] = df['Close'].rolling(10).std() * df['rsi_14']
    factors['alpha_vwapdev_over_macd'] = (df['Close'] - df['vwap']).rank(pct=True) / (df['macd_diff'] + 1e-6)
    
    # Factor 22-33: Oversold Reversals
    factors['alpha_close_delta_1d'] = -1 * df['Close'].diff()
    factors['alpha_momentum_5d_min_rank'] = -1 * (df['Close'] - df['Close'].shift(5)).rank(pct=True)
    factors['alpha_ma200_close_diff_rank'] = (df['ma200'] - df['Close']).rank(pct=True)
    factors['alpha_volatility_10d_rank_neg'] = -1 * df['Close'].rolling(10).std().rank(pct=True)
    factors['alpha_rev_rsi_times_lowdev'] = (100 - df['rsi_14']) * ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001))
    factors['alpha_volrank_times_revrsi'] = -1 * df['Close'].rolling(10).std().rank(pct=True) * (100 - df['rsi_14'])
    factors['alpha_rsi_vs_50'] = df['rsi_14'] - 50
    factors['alpha_rsi_bounce_rank'] = (((df['rsi_14'] < 30) & (df['rsi_14'] > df['rsi_14'].shift(1))).astype(int) * (df['rsi_14'] - 50)).rank(pct=True)
    factors['alpha_macd_cross'] = ((df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)).astype(float)
    factors['alpha_macd_cross_strength'] = factors['alpha_macd_cross'] * df['macd_diff']
    factors['alpha_price_vs_boll'] = ((df['Close'] < df['boll_lower']).astype(int) * df['Close'])
    factors['alpha_boll_rebound'] = ((df['Close'].shift(1) < df['boll_lower']) & (df['Close'] > df['boll_lower'])).astype(int)
    
    # Factor 34-40: Advanced Combinations
    factors['alpha_close_median_dev_ratio'] = (df['Close'] - (df['High'] + df['Low']) / 2) / (df['High'] - df['Low'] + 0.001)
    factors['alpha_macd_rank_times_rsi'] = df['macd_diff'].rank(pct=True) * df['rsi_14']
    factors['alpha_macd_rank_times_price'] = df['macd_diff'].rank(pct=True) * df['Close']
    factors['alpha_rsi_rank_times_close'] = df['rsi_14'].rank(pct=True) * df['Close']
    factors['alpha_rsi_rise_velocity'] = df['rsi_14'] - df['rsi_14'].shift(3)
    factors['alpha_boll_bandwidth'] = df['boll_upper'] - df['boll_lower']
    factors['alpha_close_boll_mid_ratio'] = (df['Close'] - df['boll_mid']) / (df['boll_upper'] - df['boll_lower'] + 1e-6)
    
    return factors

def prepare_training_data(df, factors_df, forward_window=5):
    """
    Create training samples:
    - X: 40 factors on day t
    - y_reg: forward 5-day return (clipped to [5%, 95%])
    - y_clf: binary label (1 if return > 0, 0 otherwise)
    """
    
    # Compute forward returns
    df['forward_return_5d'] = df['Close'].shift(-forward_window) / df['Close'] - 1
    
    # Clip to [5%, 95%] percentile to handle outliers
    percentile_5 = df['forward_return_5d'].quantile(0.05)
    percentile_95 = df['forward_return_5d'].quantile(0.95)
    df['forward_return_5d_clipped'] = df['forward_return_5d'].clip(percentile_5, percentile_95)
    
    # Binary label: 1 if positive return, 0 otherwise
    df['label_up_down'] = (df['forward_return_5d'] > 0).astype(int)
    
    # Combine factors with targets
    data = pd.concat([factors_df, df[['forward_return_5d_clipped', 'label_up_down']]], axis=1)
    
    # Drop NaN rows
    data = data.dropna()
    
    return data

# Z-score normalization (cross-sectional, per day)
def normalize_features(data, factor_cols):
    """Normalize factors using daily cross-sectional z-score"""
    normalized = data[factor_cols].copy()
    for date in data.index.get_level_values(0).unique():
        mask = data.index.get_level_values(0) == date
        daily_data = normalized.loc[mask]
        mean = daily_data.mean()
        std = daily_data.std()
        normalized.loc[mask] = (normalized.loc[mask] - mean) / (std + 1e-8)
    
    return normalized
```

---

## 1.4 Signal Generation & Backtesting

### **Daily Signal Generation**

```python
def generate_signals(model, factors_df, device):
    """
    Input: factors_df (N_stocks, 40 factors) for a single day
    Output: deep_alpha scores (N_stocks,) for ranking
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(factors_df.values, dtype=torch.float32).to(device)
        reg_out, clf_out = model(X)
        
        # Combine regression (magnitude) and classification (direction) into single score
        # deep_alpha = regression_output × classification_confidence
        deep_alpha = reg_out.squeeze().cpu().numpy() * clf_out.squeeze().cpu().numpy()
    
    return deep_alpha

def backtest_long_short(daily_signals, prices, initial_cash=1e6, trans_cost=0.015):
    """
    Daily backtest:
    - Long top-5 stocks by deep_alpha
    - Short bottom-5 stocks
    - Hold until next rebalance (daily)
    - Include transaction costs (0.015%) and profit tax (0.20%)
    """
    
    returns = []
    cash = initial_cash
    position_value = 0
    
    for t in range(len(daily_signals)):
        day_signals = daily_signals[t]
        day_prices = prices[t]
        
        # Rank stocks
        top_5_idx = np.argsort(day_signals)[-5:]
        bottom_5_idx = np.argsort(day_signals)[:5]
        
        # Compute P&L from yesterday's position
        if t > 0:
            prev_top_5_idx = np.argsort(daily_signals[t-1])[-5:]
            prev_prices = prices[t-1][prev_top_5_idx]
            current_prices = prices[t][prev_top_5_idx]
            pnl = (current_prices / prev_prices - 1).sum() * (cash / 10)  # 1/10 of cash per top-5
            
            # Apply profit tax (20% on gains)
            if pnl > 0:
                pnl *= 0.8
            
            cash += pnl
            position_value = cash
        
        # Transaction cost on rebalance
        trans_cost_amount = (cash * trans_cost / 100) * 2  # Buy + Sell
        cash -= trans_cost_amount
        
        returns.append((cash - initial_cash) / initial_cash)
    
    return np.array(returns)
```

### **Information Coefficient Calculation**

```python
from scipy.stats import spearmanr

def compute_ic(model_scores, realized_returns):
    """
    Spearman rank correlation between model scores and realized 5-day returns
    IC: daily metric of ranking power
    IR: IC / std(IC) - consistency over time
    """
    
    daily_ics = []
    for t in range(len(model_scores)):
        ic_t = spearmanr(model_scores[t], realized_returns[t])[0]
        daily_ics.append(ic_t)
    
    daily_ics = np.array(daily_ics)
    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)
    ir = mean_ic / (std_ic + 1e-8)
    
    return mean_ic, std_ic, ir
```

---

## Part 2: ResNet-1D on Pure OHLCV (600-day lookback) (2024)

### Overview
- **Architecture:** 1D Convolutional ResNet with skip connections
- **Input:** 600 days of OHLCV data (5×600 = 3,000 values)
- **Output:** 3-class classification (up 10%, down 10%, sideways)
- **Key Innovation:** Softmax logit thresholding for confidence filtering
- **Performance:** Korea 75% return/Sharpe 1.57, US 35% return/Sharpe 0.61

---

## 2.1 Model Architecture

### **ResNet-1D Architecture**

```python
import torch
import torch.nn as nn

class ResNet1D(nn.Module):
    def __init__(self, input_channels=5, num_classes=3, num_blocks=5, fixed_channels=12, kernel_sizes=[7, 5, 3]):
        super(ResNet1D, self).__init__()
        
        self.num_blocks = num_blocks
        self.fixed_channels = fixed_channels
        self.kernel_sizes = kernel_sizes
        
        # Initial convolution: (5, 600) → (12, 600)
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=fixed_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.initial_bn = nn.BatchNorm1d(fixed_channels)
        
        # ResNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(self._make_block())
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected + softmax
        self.fc = nn.Linear(fixed_channels, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def _make_block(self):
        """Single ResNet block: 3 conv layers with skip connection"""
        layers = []
        
        for i, k_size in enumerate(self.kernel_sizes):
            conv = nn.Conv1d(
                in_channels=self.fixed_channels,
                out_channels=self.fixed_channels,
                kernel_size=k_size,
                stride=1,
                padding=k_size // 2,
                bias=False
            )
            bn = nn.BatchNorm1d(self.fixed_channels)
            layers.append(conv)
            layers.append(bn)
            if i < len(self.kernel_sizes) - 1:
                layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, 5, 600)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = torch.relu(x)
        
        # ResNet blocks with skip connections
        for block in self.blocks:
            identity = x  # Skip connection
            x = block(x)
            x = torch.relu(x + identity)  # Add + ReLU
        
        # Global Average Pooling: (batch, 12, 600) → (batch, 12)
        x = self.gap(x).squeeze(-1)
        
        # Fully connected
        x = self.fc(x)  # (batch, 3)
        logits = x
        probs = self.softmax(x)
        
        return logits, probs  # Return both logits and probabilities
```

### **Key Design Points**
1. **Skip connections** allow lower layers to learn short-term trends (5-day MA equivalent) and higher layers to learn long-term trends (480-day MA).
2. **Fixed 12 channels** throughout reduces overfitting.
3. **Kernel sizes [7, 5, 3]** in each block progressively extract finer features.
4. **Global Average Pooling** aggregates learned features across the 600-day window.

---

## 2.2 Labeling Strategy

### **Label Definition (3-Class Classification)**

```python
def create_labels(df, target_period=5, threshold_pct=0.10):
    """
    For each row in lookback window, check if price rises/falls 10% within D days
    
    Labels:
    - 2: highest price in next D days >= close × 1.10
    - 0: lowest price in next D days <= close × 0.90
    - 1: sideways (neither hits threshold)
    """
    
    labels = []
    
    for i in range(len(df) - target_period):
        close_price = df['Close'].iloc[i]
        future_window = df['Close'].iloc[i+1:i+target_period+1]
        future_high = future_window.max()
        future_low = future_window.min()
        
        upper_threshold = close_price * (1 + threshold_pct)
        lower_threshold = close_price * (1 - threshold_pct)
        
        if future_high >= upper_threshold:
            label = 2  # UP 10%
        elif future_low <= lower_threshold:
            label = 0  # DOWN 10%
        else:
            label = 1  # SIDEWAYS
        
        labels.append(label)
    
    return np.array(labels)

# Example: Create datasets for multiple target periods
def prepare_resnet_data(df, lookback=600, target_periods=[1, 3, 5, 10, 15, 20, 30]):
    """
    Create sliding windows of 600 days OHLCV + labels for each target period
    """
    
    datasets = {}
    
    for target_d in target_periods:
        X = []  # (num_samples, 5, 600)
        y = []  # (num_samples,)
        
        for i in range(lookback, len(df) - target_d):
            # Sliding window: last 600 days
            window = df.iloc[i-lookback:i][['Open', 'High', 'Low', 'Close', 'Volume']].values
            window = window.T  # (5, 600)
            
            # Log prices to avoid exploding gradients
            window[:4] = np.log10(window[:4] + 1e-8)
            
            X.append(window)
            
            # Label
            close_price = df['Close'].iloc[i]
            future_high = df['High'].iloc[i+1:i+target_d+1].max()
            future_low = df['Low'].iloc[i+1:i+target_d+1].min()
            
            upper_threshold = close_price * 1.10
            lower_threshold = close_price * 0.90
            
            if future_high >= upper_threshold:
                y.append(2)
            elif future_low <= lower_threshold:
                y.append(0)
            else:
                y.append(1)
        
        datasets[f'target_{target_d}d'] = {
            'X': np.array(X),
            'y': np.array(y)
        }
    
    return datasets
```

---

## 2.3 Training & Validation with Softmax Thresholding

### **Training Setup**

```python
def train_resnet(model, train_loader, val_loader, epochs=50, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_resnet_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_resnet_model.pt'))
    return model
```

### **Softmax Logit Thresholding for Trading**

```python
def select_trades_by_threshold(model, test_loader, threshold=0.75, device='cuda'):
    """
    Select only stocks where max(softmax) >= threshold
    This increases confidence and profitability at the cost of fewer trades
    """
    
    model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            _, probs = model(X_batch)  # (batch, 3)
            
            max_prob, pred_class = torch.max(probs, dim=1)
            
            predictions.extend(pred_class.cpu().numpy())
            confidences.extend(max_prob.cpu().numpy())
    
    # Filter by threshold
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    high_conf_mask = confidences >= threshold
    high_conf_predictions = predictions[high_conf_mask]
    
    print(f"Threshold={threshold}: {high_conf_mask.sum()} / {len(predictions)} trades ({100*high_conf_mask.mean():.1f}%)")
    
    return high_conf_predictions, high_conf_mask

# Validate multiple thresholds
thresholds = [0.0, 0.7, 0.8, 0.9, 0.99, 0.999]
results = {}

for threshold in thresholds:
    preds, mask = select_trades_by_threshold(model, val_loader, threshold=threshold, device=device)
    
    # Compute accuracy only on high-confidence predictions
    val_y_filtered = val_y[mask]
    val_preds_filtered = preds
    
    accuracy = (val_preds_filtered == val_y_filtered).mean()
    dataset_proportion = mask.mean()
    
    results[threshold] = {
        'accuracy': accuracy,
        'dataset_proportion': dataset_proportion,
        'num_trades': mask.sum()
    }

# Plot results (inverted-U shape expected)
import matplotlib.pyplot as plt
thresholds_list = list(results.keys())
accuracies = [results[t]['accuracy'] for t in thresholds_list]
proportions = [results[t]['dataset_proportion'] for t in thresholds_list]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(thresholds_list, accuracies, 'o-')
plt.xlabel('Softmax Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Threshold (Higher = Fewer Trades, Higher Quality)')

plt.subplot(1, 2, 2)
plt.plot(thresholds_list, proportions, 's-', color='orange')
plt.xlabel('Softmax Threshold')
plt.ylabel('Dataset Proportion')
plt.title('Trading Frequency vs Threshold')
plt.tight_layout()
plt.show()
```

---

## 2.4 Backtesting with Trading Rules

### **Daily Backtest Engine**

```python
def backtest_resnet(model, prices_df, target_period=5, threshold=0.99, 
                    initial_cash=10000, entry_ratio=1/20, 
                    commission=0.015, profit_tax=0.20, device='cuda'):
    """
    Daily backtest:
    - Model scores all stocks end-of-day
    - Buy if class=2 (UP 10%) AND softmax > threshold
    - Hold until +/-10% or D days pass
    - Include realistic costs
    """
    
    model.eval()
    
    positions = []  # List of (entry_price, entry_date, exit_date, exit_reason)
    cash = initial_cash
    equity_curve = [initial_cash]
    trades = 0
    
    for day_idx in range(600, len(prices_df) - target_period):
        # Prepare OHLCV window for this day
        window = prices_df.iloc[day_idx-600:day_idx][['Open', 'High', 'Low', 'Close', 'Volume']].values
        window = window.T  # (5, 600)
        window[:4] = np.log10(window[:4] + 1e-8)
        
        # Model prediction
        X = torch.tensor(window[np.newaxis, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, probs = model(X)
            max_prob = probs[0].max().item()
            pred_class = probs[0].argmax().item()
        
        # Trading decision
        if pred_class == 2 and max_prob >= threshold:  # BUY signal
            entry_price = prices_df['Close'].iloc[day_idx]
            entry_cash = cash * entry_ratio
            shares = entry_cash / entry_price
            cash -= entry_cash * (1 + commission / 100)
            
            positions.append({
                'entry_price': entry_price,
                'shares': shares,
                'entry_day': day_idx,
                'exit_price': None,
                'exit_day': None,
                'pnl': 0
            })
            trades += 1
        
        # Check exit conditions for open positions
        for pos in positions:
            if pos['exit_price'] is not None:
                continue  # Already closed
            
            current_price = prices_df['Close'].iloc[day_idx]
            days_held = day_idx - pos['entry_day']
            
            # Exit conditions: +/-10% profit/loss OR target_period days
            price_change_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            if price_change_pct >= 0.10:  # +10%
                pos['exit_price'] = pos['entry_price'] * 1.10
                pos['exit_day'] = day_idx
                pnl = pos['shares'] * pos['exit_price']
                pnl *= (1 - profit_tax)  # Apply tax on gains
                cash += pnl
            elif price_change_pct <= -0.10:  # -10%
                pos['exit_price'] = pos['entry_price'] * 0.90
                pos['exit_day'] = day_idx
                pnl = pos['shares'] * pos['exit_price']
                cash += pnl
            elif days_held >= target_period:  # Time stop
                pos['exit_price'] = current_price
                pos['exit_day'] = day_idx
                pnl = pos['shares'] * current_price
                if pnl > pos['shares'] * pos['entry_price']:  # Gain
                    pnl *= (1 - profit_tax)
                cash += pnl
        
        # Update equity
        open_pos_value = sum([p['shares'] * prices_df['Close'].iloc[day_idx] 
                               for p in positions if p['exit_price'] is None])
        equity = cash + open_pos_value
        equity_curve.append(equity)
    
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] - initial_cash) / initial_cash
    max_drawdown = (equity_curve.min() - initial_cash) / initial_cash
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    return {
        'equity_curve': equity_curve,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': trades,
        'data': {
            'returns': daily_returns,
            'equity': equity_curve
        }
    }
```

---

## 2.5 Model Selection: F1 Macro Scoring

### **Choosing Best Model & Threshold**

```python
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate_all_models_thresholds(model, val_loader, thresholds=[0.0, 0.7, 0.8, 0.9, 0.99]):
    """
    Rank models by F1 macro score across all thresholds.
    Select the threshold that maximizes F1 for each label, with dataset proportion > 0.00001
    """
    
    model.eval()
    all_probs = []
    all_labels = []
    
    # Collect all predictions
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            _, probs = model(X_batch.to(device))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    results_summary = []
    
    for threshold in thresholds:
        max_prob = all_probs.max(axis=1)
        pred_labels = all_probs.argmax(axis=1)
        
        # Mask: keep only high-confidence predictions
        mask = max_prob >= threshold
        dataset_prop = mask.mean()
        
        if dataset_prop > 0:
            y_filtered = all_labels[mask]
            pred_filtered = pred_labels[mask]
            
            accuracy = accuracy_score(y_filtered, pred_filtered)
            f1 = f1_score(y_filtered, pred_filtered, average='macro')
            precision_macro = precision_score(y_filtered, pred_filtered, average='macro', zero_division=0)
            recall_macro = recall_score(y_filtered, pred_filtered, average='macro', zero_division=0)
        else:
            accuracy = np.nan
            f1 = np.nan
            precision_macro = np.nan
            recall_macro = np.nan
        
        results_summary.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'f1_macro': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'dataset_proportion': dataset_prop
        })
    
    results_df = pd.DataFrame(results_summary)
    
    # Select best threshold: maximize F1 macro with sufficient dataset proportion
    valid_results = results_df[results_df['dataset_proportion'] > 0.00001]
    best_threshold = valid_results.loc[valid_results['f1_macro'].idxmax(), 'threshold']
    
    print(results_df)
    print(f"\nBest Threshold: {best_threshold}")
    
    return best_threshold, results_df
```

---

## Comparison Summary

| Aspect | Dual-Task MLP (Behavioral Factors) | ResNet-1D (OHLCV) |
|--------|-------|-------|
| **Input** | 40 engineered behavioral factors | Raw OHLCV (5×600) |
| **Architecture** | MLP 64→32→(reg+clf heads) | ResNet-1D, 5 blocks, 12 channels |
| **Output** | 1 continuous + 1 binary | 3-class classification |
| **Training Data** | Feature engineering (2–3 hours setup) | Direct preprocessing (minutes) |
| **Interpretability** | SHAP factor attribution | Black-box; visual chart patterns |
| **Sharpe Ratio** | 1.61 (top-5 long/short) | 1.57 (Korea), 0.61 (US) |
| **Expected Performance Live** | Sharpe 0.8–1.2 (30–40% worse) | Sharpe 0.9–1.2 (Korea); 0.4–0.5 (US) |
| **Robustness** | More robust; behavioral foundations | Less robust to regime changes |
| **Ease of Implementation** | Moderate; needs factor library | Very easy; pure OHLCV |

---

## Implementation Roadmap for Your Paper-Trader

### **Phase 1 (Week 1–2): Data Preparation**
1. Download OHLCV data from yfinance or EODData
2. For **Dual-Task MLP:** Implement 40 factor computation (use the formulas provided)
3. For **ResNet-1D:** Create sliding windows of 600-day OHLCV

### **Phase 2 (Week 3–4): Model Training**
1. Build and train model on 2015–2019 data
2. Validate on 2020–2021
3. Test on 2022–2024 (recent, unseen)

### **Phase 3 (Week 5–6): Backtesting & Optimization**
1. Implement daily backtest with realistic costs
2. Optimize threshold (for ResNet) or feature selection (for MLP)
3. Compute IC, Sharpe, max drawdown metrics

### **Phase 4 (Week 7+): Documentation & Integration**
1. Write clear README with strategy description
2. Add backtesting results (plots, metrics table)
3. Create simple unit tests
4. Integrate into your paper-trader framework

---

*Last Updated: December 2025*