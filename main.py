"""
Commodity Market Prediction System
===================================
Hybrid Architecture: CNN + LSTM + Transformer for price forecasting
Sentiment Analysis: FinBERT-based geopolitical news analysis
Reinforcement Learning: PPO-based trade execution agent
Case Study: Gold (XAU/USD) under geopolitical stress conditions

Requirements:
    pip install torch transformers pandas numpy scikit-learn yfinance stable-baselines3 gym matplotlib seaborn
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math

# ─────────────────────────────────────────────
# SECTION 1: DATA COLLECTION & PREPROCESSING
# ─────────────────────────────────────────────

class OHLCVDataLoader:
    """
    Loads and preprocesses OHLCV (Open, High, Low, Close, Volume) data.
    Uses yfinance to fetch historical gold prices (GC=F).
    Falls back to synthetic data if network unavailable.
    """

    def __init__(self, ticker="GC=F", start="2018-01-01", end="2024-01-01"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            import yfinance as yf
            print(f"[DATA] Fetching {self.ticker} from {self.start} to {self.end}...")
            df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            print(f"[DATA] Fetched {len(df)} rows.")
            return df
        except Exception as e:
            print(f"[DATA] yfinance unavailable ({e}). Using synthetic data.")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate realistic synthetic gold OHLCV data for offline testing."""
        np.random.seed(42)
        dates = pd.date_range(start=self.start, end=self.end, freq="B")
        n = len(dates)
        # Simulate gold price walk starting ~$1300
        returns = np.random.normal(0.0002, 0.012, n)
        close = 1300 * np.exp(np.cumsum(returns))
        high  = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low   = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = close * (1 + np.random.normal(0, 0.003, n))
        vol   = np.random.randint(50000, 200000, n).astype(float)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                           "Close": close, "Volume": vol}, index=dates)
        print(f"[DATA] Generated {n} synthetic rows.")
        return df

    def add_technical_indicators(self, df):
        """Add RSI, MACD, Bollinger Bands, ATR as additional features."""
        # RSI (14-period)
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

        # Bollinger Bands
        sma20       = df["Close"].rolling(20).mean()
        std20       = df["Close"].rolling(20).std()
        df["BB_upper"] = sma20 + 2 * std20
        df["BB_lower"] = sma20 - 2 * std20
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (sma20 + 1e-9)

        # ATR (Average True Range)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()

        # Log returns
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        return df.dropna()

    def prepare_sequences(self, df, seq_len=60, target_col="Close"):
        """
        Create overlapping sequences of length `seq_len` for time-series modeling.
        Returns: X (samples, seq_len, features), y (samples,)
        """
        feature_cols = [c for c in df.columns if c != target_col]
        data = df[feature_cols + [target_col]].values

        # Fit scaler on all columns
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(seq_len, len(data_scaled)):
            X.append(data_scaled[i - seq_len:i, :-1])   # all features
            y.append(data_scaled[i, -1])                 # target: Close

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# SECTION 2: CNN-LSTM-TRANSFORMER ARCHITECTURE
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNLSTMTransformer(nn.Module):
    """
    Hybrid architecture combining:
    1. CNN  — extracts local temporal patterns (short-term momentum)
    2. LSTM — captures sequential dependencies (trend memory)
    3. Transformer encoder — models long-range attention across time steps
    4. Fully connected head — outputs single price prediction
    """

    def __init__(self, input_size, seq_len=60,
                 cnn_filters=64, cnn_kernel=3,
                 lstm_hidden=128, lstm_layers=2,
                 d_model=128, nhead=4, num_encoder_layers=2,
                 dropout=0.2):
        super().__init__()
        self.seq_len    = seq_len
        self.input_size = input_size

        # ── CNN Block ──────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── LSTM Block ─────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # ── Transformer Block ──────────────────────────
        self.input_proj = nn.Linear(lstm_hidden, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ── Output Head ────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)

        # CNN expects (batch, features, seq_len)
        cnn_out = self.cnn(x.permute(0, 2, 1))           # (B, cnn_filters, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)                # (B, seq_len, cnn_filters)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)                  # (B, seq_len, lstm_hidden)

        # Transformer
        t_in  = self.input_proj(lstm_out)                 # (B, seq_len, d_model)
        t_in  = self.pos_enc(t_in)
        t_out = self.transformer(t_in)                    # (B, seq_len, d_model)

        # Use last time step for prediction
        out = self.fc(t_out[:, -1, :])                    # (B, 1)
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# SECTION 3: FINBERT SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

class SentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for geopolitical news headlines.
    Falls back to simple lexicon-based scoring if transformers unavailable.
    Outputs sentiment scores: positive (bullish), negative (bearish), neutral.
    """

    def __init__(self, use_finbert=True):
        self.use_finbert = use_finbert
        self.model      = None
        self.tokenizer  = None

        if use_finbert:
            self._load_finbert()

    def _load_finbert(self):
        """Load ProsusAI/finbert from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            print("[SENTIMENT] Loading FinBERT...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            print("[SENTIMENT] FinBERT loaded.")
        except Exception as e:
            print(f"[SENTIMENT] FinBERT unavailable ({e}). Using lexicon fallback.")
            self.use_finbert = False

    def analyze(self, headlines: list) -> pd.DataFrame:
        """
        Analyze a list of headlines.
        Returns DataFrame with columns: headline, positive, negative, neutral, score
        score = positive - negative  (ranges from -1 to +1)
        """
        results = []
        for h in headlines:
            if self.use_finbert and self.model:
                score_dict = self._finbert_score(h)
            else:
                score_dict = self._lexicon_score(h)
            score_dict["headline"] = h
            score_dict["score"]    = score_dict["positive"] - score_dict["negative"]
            results.append(score_dict)
        return pd.DataFrame(results)

    def _finbert_score(self, text):
        """Run FinBERT inference on a single text."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # FinBERT label order: positive, negative, neutral
        return {"positive": probs[0], "negative": probs[1], "neutral": probs[2]}

    def _lexicon_score(self, text):
        """Simple keyword-based fallback sentiment."""
        positive_words = ["surge", "rally", "gain", "rise", "bullish", "safe haven",
                          "crisis", "geopolitical", "war", "inflation", "uncertainty"]
        negative_words = ["drop", "fall", "decline", "bearish", "peace", "recovery",
                          "stable", "calm", "rate hike", "strong dollar"]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower) / max(len(positive_words), 1)
        neg = sum(1 for w in negative_words if w in text_lower) / max(len(negative_words), 1)
        neu = max(0.0, 1.0 - pos - neg)
        return {"positive": pos, "negative": neg, "neutral": neu}

    def generate_sample_news(self, n=50) -> list:
        """Generate realistic geopolitical gold-market headlines for demo."""
        headlines = [
            "Gold surges as geopolitical tensions escalate in Middle East",
            "Federal Reserve signals rate hike amid persistent inflation",
            "Ukraine conflict drives investors toward safe-haven assets",
            "US dollar strengthens, pressuring gold prices lower",
            "OPEC+ cuts production, inflation fears push gold higher",
            "China increases gold reserves amid US-China trade tensions",
            "Gold falls as risk appetite returns on ceasefire hopes",
            "Central banks globally increase gold holdings in 2023",
            "Inflation data beats expectations, gold rallies sharply",
            "IMF warns of global recession risk, gold demand spikes",
            "Russia sanctions impact commodity markets broadly",
            "Gold hits all-time high amid banking sector stress",
            "Strong jobs report reduces safe-haven demand for gold",
            "Geopolitical risk index reaches decade high, gold climbs",
            "Dollar weakness boosts gold to multi-month highs",
        ] * (n // 15 + 1)
        return headlines[:n]


# ─────────────────────────────────────────────
# SECTION 4: PPO REINFORCEMENT LEARNING AGENT
# ─────────────────────────────────────────────

class TradingEnvironment:
    """
    Custom trading environment for gold market.
    State:  price features + portfolio state + sentiment score
    Action: 0=Hold, 1=Buy, 2=Sell
    Reward: risk-adjusted PnL (Sharpe-like)
    """

    def __init__(self, prices: np.ndarray, sentiments: np.ndarray = None,
                 initial_cash=100_000, transaction_cost=0.001):
        self.prices           = prices
        self.sentiments       = sentiments if sentiments is not None else np.zeros(len(prices))
        self.initial_cash     = initial_cash
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.cash        = self.initial_cash
        self.position    = 0          # units of gold held
        self.step_idx    = 10         # start after burn-in
        self.portfolio_v = [self.initial_cash]
        self.returns_log = []
        return self._get_state()

    def _get_state(self):
        """State: last 10 normalized returns + current position + sentiment."""
        window    = self.prices[max(0, self.step_idx-10): self.step_idx]
        if len(window) < 2:
            rets = np.zeros(10)
        else:
            rets = np.diff(window) / (window[:-1] + 1e-9)
            rets = np.pad(rets, (10 - len(rets), 0))
        pos_norm    = np.array([self.position / 10.0])
        sentiment   = np.array([self.sentiments[min(self.step_idx, len(self.sentiments)-1)]])
        return np.concatenate([rets, pos_norm, sentiment]).astype(np.float32)

    def step(self, action):
        """Execute action; return (next_state, reward, done, info)."""
        price = self.prices[self.step_idx]
        prev_value = self.cash + self.position * self.prices[self.step_idx - 1]

        if action == 1 and self.cash >= price:       # BUY
            units = int(self.cash * 0.95 / price)
            cost  = units * price * (1 + self.transaction_cost)
            self.cash     -= cost
            self.position += units

        elif action == 2 and self.position > 0:      # SELL
            proceeds       = self.position * price * (1 - self.transaction_cost)
            self.cash     += proceeds
            self.position  = 0

        # Portfolio value after action
        curr_value = self.cash + self.position * price
        ret        = (curr_value - prev_value) / (prev_value + 1e-9)
        self.returns_log.append(ret)
        self.portfolio_v.append(curr_value)

        # Reward: return minus volatility penalty (promotes Sharpe)
        vol    = np.std(self.returns_log[-20:]) if len(self.returns_log) >= 2 else 0
        reward = ret - 0.5 * vol

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1
        return self._get_state(), reward, done, {"portfolio": curr_value}


class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization (PPO) agent.
    Actor-Critic architecture for trade execution decisions.
    """

    def __init__(self, state_dim=12, action_dim=3, hidden=128):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Actor: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: estimates state value
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        enc   = self.encoder(state)
        probs = self.actor(enc)
        value = self.critic(enc)
        return probs, value

    def select_action(self, state):
        """Sample action from policy distribution."""
        state  = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.forward(state)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def train_ppo(self, env, n_episodes=50, gamma=0.99, clip_eps=0.2, lr=3e-4):
        """
        Core PPO training loop with clipped surrogate objective.
        Alternates between environment rollout and policy update.
        """
        optimizer     = optim.Adam(self.parameters(), lr=lr)
        episode_rewards = []

        for ep in range(n_episodes):
            state        = env.reset()
            done         = False
            ep_reward    = 0

            # Collect trajectory
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done:
                action, log_p, val = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                states.append(state); actions.append(action)
                rewards.append(reward); log_probs.append(log_p)
                values.append(val.squeeze())
                ep_reward += reward
                state = next_state

            episode_rewards.append(ep_reward)

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # PPO update (simplified single epoch)
            log_probs_t  = torch.stack(log_probs)
            values_t     = torch.stack(values)
            advantages   = (returns - values_t.detach())

            states_t  = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)
            new_probs, new_vals = self.forward(states_t)
            dist_new  = torch.distributions.Categorical(new_probs)
            new_log_p = dist_new.log_prob(actions_t)

            ratio     = torch.exp(new_log_p - log_probs_t.detach())
            surr1     = ratio * advantages
            surr2     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_vals.squeeze(), returns)
            loss        = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()

            if (ep + 1) % 10 == 0:
                print(f"  [PPO] Episode {ep+1}/{n_episodes} | Avg Reward: {np.mean(episode_rewards[-10:]):.4f}")

        return episode_rewards


# ─────────────────────────────────────────────
# SECTION 5: TRAINING PIPELINE
# ─────────────────────────────────────────────

class CommodityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_price_model(model, X_train, y_train, X_val, y_val,
                      epochs=50, batch_size=64, lr=1e-3, device="cpu"):
    """Train the CNN-LSTM-Transformer price prediction model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_ds  = CommodityDataset(X_train, y_train)
    val_ds    = CommodityDataset(X_val,   y_val)
    train_dl  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl    = DataLoader(val_ds,   batch_size=batch_size)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                val_loss += criterion(pred, yb).item()

        train_losses.append(epoch_loss / len(train_dl))
        val_losses.append(val_loss / len(val_dl))
        scheduler.step(val_losses[-1])

        if (epoch + 1) % 10 == 0:
            print(f"  [PRICE MODEL] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

    return train_losses, val_losses


# ─────────────────────────────────────────────
# SECTION 6: BACKTESTING & PERFORMANCE METRICS
# ─────────────────────────────────────────────

class Backtester:
    """
    Backtests the combined system (price model + sentiment + RL agent)
    against historical gold prices. Computes key performance metrics.
    """

    def __init__(self, prices: np.ndarray, predictions: np.ndarray,
                 sentiments: np.ndarray = None, initial_capital=100_000):
        self.prices          = prices
        self.predictions     = predictions
        self.sentiments      = sentiments if sentiments is not None else np.zeros(len(prices))
        self.initial_capital = initial_capital

    def run_strategy(self):
        """
        Strategy logic:
        - Buy  when predicted price > current price AND sentiment > 0.1
        - Sell when predicted price < current price OR sentiment < -0.1
        - Hold otherwise
        """
        cash      = self.initial_capital
        position  = 0
        portfolio = []
        trades    = []
        tc        = 0.001  # 0.1% transaction cost

        for i in range(len(self.predictions)):
            price   = self.prices[i]
            pred    = self.predictions[i]
            senti   = self.sentiments[min(i, len(self.sentiments)-1)]
            pv      = cash + position * price
            portfolio.append(pv)

            if pred > price * 1.005 and senti > 0.0 and cash >= price:
                units      = int(cash * 0.9 / (price * (1 + tc)))
                if units > 0:
                    cash      -= units * price * (1 + tc)
                    position  += units
                    trades.append(("BUY", i, price, units))

            elif (pred < price * 0.995 or senti < -0.05) and position > 0:
                cash     += position * price * (1 - tc)
                trades.append(("SELL", i, price, position))
                position  = 0

        # Final liquidation
        if position > 0:
            cash += position * self.prices[-1] * (1 - tc)
            portfolio[-1] = cash

        return np.array(portfolio), trades

    def compute_metrics(self, portfolio: np.ndarray) -> dict:
        """Compute Sharpe Ratio, Max Drawdown, CAGR, Win Rate, Calmar Ratio."""
        returns = np.diff(portfolio) / (portfolio[:-1] + 1e-9)

        # Sharpe Ratio (annualized, assumes daily returns)
        sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)

        # Maximum Drawdown
        peak      = np.maximum.accumulate(portfolio)
        drawdown  = (portfolio - peak) / (peak + 1e-9)
        max_dd    = drawdown.min()

        # CAGR
        n_years   = len(portfolio) / 252
        cagr      = (portfolio[-1] / portfolio[0]) ** (1 / max(n_years, 0.01)) - 1

        # Calmar Ratio
        calmar    = cagr / (abs(max_dd) + 1e-9)

        # Total Return
        total_ret = (portfolio[-1] - portfolio[0]) / portfolio[0]

        # Volatility (annualized)
        vol       = returns.std() * np.sqrt(252)

        return {
            "Total Return (%)":      round(total_ret * 100, 2),
            "CAGR (%)":              round(cagr * 100, 2),
            "Sharpe Ratio":          round(sharpe, 4),
            "Max Drawdown (%)":      round(max_dd * 100, 2),
            "Annualized Volatility": round(vol * 100, 2),
            "Calmar Ratio":          round(calmar, 4),
            "Final Portfolio ($)":   round(portfolio[-1], 2),
        }


# ─────────────────────────────────────────────
# SECTION 7: VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(df_prices, portfolio, train_losses, val_losses,
                 metrics: dict, save_path="results.png"):
    """Generate comprehensive performance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Commodity Market Prediction — Gold (XAU/USD)", fontsize=16, y=1.01)

    # 1. Gold price history
    ax1 = axes[0, 0]
    ax1.plot(df_prices.index[-len(portfolio):], df_prices["Close"].values[-len(portfolio):],
             color="#FFD700", linewidth=1.5, label="Actual Gold Price")
    ax1.set_title("Gold Price History"); ax1.set_ylabel("Price (USD)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. Portfolio value
    ax2 = axes[0, 1]
    ax2.plot(portfolio, color="#2196F3", linewidth=1.5)
    ax2.axhline(portfolio[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax2.set_title("Portfolio Value Over Time"); ax2.set_ylabel("Value (USD)")
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. Training loss curves
    ax3 = axes[1, 0]
    ax3.plot(train_losses, label="Train Loss", color="#E53935")
    ax3.plot(val_losses,   label="Val Loss",   color="#43A047")
    ax3.set_title("CNN-LSTM-Transformer Training Loss")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE Loss")
    ax3.legend(); ax3.grid(alpha=0.3)

    # 4. Metrics bar chart
    ax4 = axes[1, 1]
    display_metrics = {
        "Total Return (%)": metrics["Total Return (%)"],
        "CAGR (%)":         metrics["CAGR (%)"],
        "Sharpe Ratio":     metrics["Sharpe Ratio"],
        "Max Drawdown (%)": metrics["Max Drawdown (%)"],
        "Calmar Ratio":     metrics["Calmar Ratio"],
    }
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in display_metrics.values()]
    ax4.barh(list(display_metrics.keys()), list(display_metrics.values()), color=colors)
    ax4.set_title("Performance Metrics"); ax4.axvline(0, color="black", linewidth=0.8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[VIZ] Saved results to {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# SECTION 8: MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  COMMODITY MARKET PREDICTION SYSTEM")
    print("  Hybrid CNN-LSTM-Transformer + FinBERT + PPO")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] Using: {device}\n")

    # ── Step 1: Load & preprocess OHLCV data ──────────────
    print("[STEP 1] Data Loading & Feature Engineering")
    loader = OHLCVDataLoader(ticker="GC=F", start="2018-01-01", end="2024-01-01")
    df     = loader.fetch_data()
    df     = loader.add_technical_indicators(df)
    X, y   = loader.prepare_sequences(df, seq_len=60)

    # Train / Val / Test split (70/15/15)
    n       = len(X)
    i_val   = int(n * 0.70)
    i_test  = int(n * 0.85)
    X_train, y_train = X[:i_val],        y[:i_val]
    X_val,   y_val   = X[i_val:i_test],  y[i_val:i_test]
    X_test,  y_test  = X[i_test:],       y[i_test:]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    # ── Step 2: Train price prediction model ──────────────
    print("[STEP 2] Training CNN-LSTM-Transformer Price Model")
    input_size  = X_train.shape[2]
    price_model = CNNLSTMTransformer(input_size=input_size, seq_len=60)
    print(f"  Model parameters: {sum(p.numel() for p in price_model.parameters()):,}")

    train_losses, val_losses = train_price_model(
        price_model, X_train, y_train, X_val, y_val,
        epochs=30, batch_size=64, lr=1e-3, device=device
    )
    print()

    # ── Step 3: Sentiment Analysis ─────────────────────────
    print("[STEP 3] FinBERT Sentiment Analysis")
    analyzer  = SentimentAnalyzer(use_finbert=True)
    headlines = analyzer.generate_sample_news(n=100)
    senti_df  = analyzer.analyze(headlines)
    avg_score = senti_df["score"].mean()
    print(f"  Analyzed {len(headlines)} headlines | Avg Sentiment: {avg_score:.3f}")
    print(f"  Bullish: {(senti_df['score']>0).sum()} | Bearish: {(senti_df['score']<0).sum()}\n")

    # ── Step 4: Generate price predictions on test set ────
    print("[STEP 4] Generating Test Set Predictions")
    price_model.eval()
    with torch.no_grad():
        X_test_t  = torch.FloatTensor(X_test).to(device)
        preds_sc  = price_model(X_test_t).cpu().numpy()

    # Inverse-transform predictions & actuals
    n_features  = X_test.shape[2] + 1
    dummy_preds = np.zeros((len(preds_sc), n_features))
    dummy_preds[:, -1] = preds_sc
    dummy_actuals       = np.zeros((len(y_test), n_features))
    dummy_actuals[:, -1] = y_test

    preds_real   = loader.scaler.inverse_transform(dummy_preds)[:, -1]
    actuals_real = loader.scaler.inverse_transform(dummy_actuals)[:, -1]

    mae  = mean_absolute_error(actuals_real, preds_real)
    rmse = np.sqrt(mean_squared_error(actuals_real, preds_real))
    mape = np.mean(np.abs((actuals_real - preds_real) / (actuals_real + 1e-9))) * 100
    print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%\n")

    # ── Step 5: PPO Reinforcement Learning ─────────────────
    print("[STEP 5] Training PPO Trade Execution Agent")
    sentiments_arr = np.full(len(actuals_real), avg_score)
    env            = TradingEnvironment(actuals_real, sentiments_arr)
    ppo_agent      = PPOAgent(state_dim=12, action_dim=3)
    ep_rewards     = ppo_agent.train_ppo(env, n_episodes=30)
    print()

    # ── Step 6: Backtesting ─────────────────────────────────
    print("[STEP 6] Backtesting Strategy")
    backtester  = Backtester(actuals_real, preds_real, sentiments_arr)
    portfolio, trades = backtester.run_strategy()
    metrics     = backtester.compute_metrics(portfolio)

    print("\n  ── PERFORMANCE METRICS ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print()

    # ── Step 7: Visualize ───────────────────────────────────
    print("[STEP 7] Generating Visualizations")
    df_test = df.iloc[-len(actuals_real):]
    plot_results(df_test, portfolio, train_losses, val_losses, metrics)

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SYSTEM COMPLETE")
    print(f"  Trades executed:    {len(trades)}")
    print(f"  Final portfolio:    ${metrics['Final Portfolio ($)']:,.2f}")
    print(f"  Sharpe Ratio:       {metrics['Sharpe Ratio']}")
    print(f"  Max Drawdown:       {metrics['Max Drawdown (%)']}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
