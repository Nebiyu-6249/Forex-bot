# Forex Trading Bot

A modular Python framework for building, backtesting, and deploying automated Forex trading strategies integrated with MetaTrader 5 (MT5). The plug-in architecture lets you define a strategy in a single Python file, test it against historical data, and deploy it live — all using the same core logic.

## How It Works

```
Strategy Definition → Backtesting → Evaluation → Live Deployment
```

1. **Define a strategy** in a single Python file (plug-in style)
2. **Backtest** against historical data using the shared core engine
3. **Evaluate** performance with metrics and analytics
4. **Deploy live** to MetaTrader 5 using the same logic — no code changes needed

## Features

- Modular plug-in architecture: one file per strategy
- Shared backtesting and live deployment core logic
- MetaTrader 5 integration for real-time market data and execution
- Historical data backtesting with performance metrics
- Designed for rapid iteration: build, test, deploy, repeat

## Tech Stack

- **Language:** Python
- **Trading Platform:** MetaTrader 5 (MT5)
- **Architecture:** Modular plug-in system
- **Data:** Historical and real-time market data via MT5

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Nebiyu-6249/Forex-bot.git
cd Forex-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and configure MetaTrader 5

4. Create your strategy file following the plug-in template

5. Run backtesting or deploy live

## Architecture

The modular design separates concerns:
- **Strategy files** — define entry/exit logic in isolated Python files
- **Core engine** — handles backtesting, execution, and MT5 communication
- **Data layer** — manages historical and live data feeds

This separation means you can iterate on strategies without touching the infrastructure.

## Author

**Nebiyu Gemedu** — AI Developer | [GitHub](https://github.com/Nebiyu-6249)
