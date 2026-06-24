# Market-Advisor-Bot
 Autonomous AI Agents for Commodity Market
# Commodity Market Prediction System
## Setup & Usage Guide

### 1. Install Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn yfinance stable-baselines3 gymnasium matplotlib seaborn
```

### 2. Run the System
```bash
python main.py
```

### 3. What the System Does

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Price Forecasting | CNN + LSTM + Transformer | Predicts next-day gold close price |
| Sentiment Analysis | FinBERT (ProsusAI/finbert) | Scores geopolitical news headlines |
| Trade Execution | PPO (Reinforcement Learning) | Decides Buy / Sell / Hold |
| Backtesting | Custom engine | Simulates portfolio on test data |
| Metrics | Sharpe, Drawdown, CAGR | Evaluates strategy performance |

### 4. Data Source
- Live: Yahoo Finance via `yfinance` (ticker: `GC=F` = Gold Futures)
- Offline: Synthetic OHLCV data auto-generated if network unavailable

### 5. Output
- Console logs with training progress and metrics
- `results.png` — 4-panel visualization chart

### 6. Customization
- Change ticker to `CL=F` (crude oil), `SI=F` (silver), etc.
- Adjust `seq_len`, `epochs`, `n_episodes` for performance vs speed
- Swap FinBERT for domain-specific models (e.g., BloombergGPT embeddings)
- Connect real news APIs (GDELT, NewsAPI) for live sentiment feeds

### 7. Hardware Notes
- CPU: ~5-10 mins for full run (30 epochs + 30 PPO episodes)
- GPU (CUDA): ~1-2 mins
- RAM: minimum 4GB recommended
