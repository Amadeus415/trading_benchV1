# Trading Bench: Project Vision

## Why I am building this

I want to build a high-quality finance evaluation that tests whether an AI model can make coherent investment decisions over time—not just answer isolated finance questions.

Each model begins with **$1,000 in fake money** and can trade across multiple markets:

- Stocks
- Cryptocurrency
- Prediction markets

The evaluation runs over a defined time period, such as 2025–2026. During that period, the model must research opportunities, allocate its limited capital, manage open positions, react to changing conditions, and live with the consequences of its earlier decisions.

The goal is not simply to find the model with the highest return. The benchmark should reveal how models reason under uncertainty, manage risk, use tools, maintain context, and behave when markets move against them.

## The core idea

Most model evaluations are short, static, and disconnected from the real consequences of previous actions. Trading creates a more demanding test because decisions are sequential and stateful:

1. A model receives the same starting capital, market access, information, and rules as every other model.
2. It decides what to buy, sell, or hold.
3. A deterministic simulator executes valid orders and maintains the portfolio ledger.
4. The model returns over time to inspect its positions and make new decisions.
5. The system evaluates both the final outcome and the behavior that produced it.

All trading is simulated. Models may act autonomously inside the evaluation, but they never receive authority over real money or a brokerage account.

## What I want to learn

Trading Bench should help answer questions such as:

- Can a model form and follow a consistent investment strategy?
- Can it allocate a small amount of capital across very different asset classes?
- Does it understand position sizing, concentration, liquidity, and downside risk?
- Can it distinguish a strong thesis from a confident-sounding guess?
- Does it adapt intelligently when new information arrives?
- Can it keep track of its own decisions and explain why its view changed?
- Do cheaper models behave meaningfully differently from more capable models?
- Are good results repeatable, or mostly the product of luck?

Performance matters, but it should be evaluated alongside risk, consistency, rule compliance, reasoning quality, and reproducibility.

## MVP

The first version should be deliberately small, inexpensive, and easy to understand.

The MVP will provide:

- A fixed $1,000 paper portfolio
- A clearly defined evaluation window
- A small, explicit universe of stocks, crypto assets, and prediction markets
- Historical or replayable market data so runs can be reproduced
- A simple model interface for research, decisions, and structured orders
- Deterministic validation, execution, fees, and portfolio accounting
- A persistent record of prompts, reasoning summaries, decisions, trades, and portfolio state
- A basic dashboard or report showing positions, cash, returns, drawdown, and activity over time
- A small evaluation suite for comparing models and repeated runs

The initial system should be cheap enough to test with models such as Luna or Terra. That constraint is useful: it encourages short prompts, explicit state, understandable workflows, and an architecture that can be inspected without specialized infrastructure.

## What makes the evaluation credible

For this to be more than a trading demo, the benchmark needs clear rules and trustworthy accounting.

The system—not the model—must own consequential mechanics such as:

- Which assets are available at each point in time
- What information the model is allowed to see
- Whether an order is valid
- How orders are priced and filled
- Fees, slippage, and market-resolution rules
- Cash, holdings, profit and loss, and portfolio valuation
- Prevention of look-ahead bias and accidental future-data leakage

Every run should be reproducible and auditable. A result should be traceable from the model's inputs and decisions through execution and final scoring.

## How success should be measured

A useful scorecard should include more than raw profit:

- Total and risk-adjusted return
- Maximum drawdown and volatility
- Capital concentration and diversification
- Turnover, fees, and trading frequency
- Rule violations and invalid actions
- Quality and consistency of stated theses
- Calibration: whether confidence matches outcomes
- Responsiveness to new information
- Stability across repeated runs and different market periods
- Cost and latency of running the model

The benchmark should make it easy to inspect both the leaderboard and the story behind each result.

## Design principles

- **Fake money only.** Autonomous behavior stays inside deterministic paper accounting.
- **Simple before clever.** The entire MVP should be understandable end to end.
- **Comparable.** Models receive consistent rules, tools, data, and budgets.
- **Reproducible.** Historical runs can be replayed and independently checked.
- **Auditable.** Every decision and portfolio change has a clear record.
- **Honest about uncertainty.** Results distinguish skill, risk, cost, and luck.
- **Easy to visualize.** Someone should be able to understand a run without reading the codebase.
- **Built to grow.** The MVP establishes clean boundaries that later autonomous capabilities can use.

## Long-term direction: a trading operating system

Long term, I want to explore how this benchmark could become a self-operating trading research system.

Inspired by agent systems such as Hermes Agent and Gary Tain's gbrain, the system could eventually run asynchronously, maintain memory, schedule its own work, monitor open positions, revisit old theses, gather new evidence, and decide when action is warranted.

Possible future capabilities include:

- Persistent strategy and market memory
- Scheduled and event-driven research
- Automatic monitoring of positions and catalysts
- Specialized research, risk, execution, and review agents
- Self-critique and independent checks before trades
- Experiment tracking across strategies, prompts, models, and time periods
- Rich visual timelines explaining what the system knew and why it acted
- Long-running paper portfolios that operate without a human in the loop

The benchmark is the foundation for that vision. Before building a complex autonomous trading system, I want a controlled environment that can prove whether its behavior is competent, safe, understandable, and genuinely improving.

## The project I want this to become

I want Trading Bench to be a serious, well-designed side project—not a superficial AI wrapper or a cherry-picked profit chart. It should be technically credible, visually clear, and interesting enough that another engineer, researcher, or hiring manager can inspect it and understand both the ambition and the rigor behind it.

The first milestone is modest: build the smallest trustworthy system that can give inexpensive models $1,000 in simulated capital, let them trade over time, and show exactly how well they performed and why.

From there, the project can grow into a practical laboratory for evaluating autonomous financial agents—and eventually into the foundation of a fully self-operating paper-trading OS.
