# Learning the project — finance, trading, and a bit of ML

A guide for someone who built this app (or is reading the code) but doesn't
have a finance background. The goal is to make every concept this project
relies on understandable from first principles, with pointers to deeper
resources.

> **How to read this**: Part 1 (finance fundamentals) is the bedrock. Read
> it linearly. Parts 2–6 build on it. Skip Part 7 (glossary) until you need
> it. The resources at the end are deliberately curated — the financial
> world has a lot of bad content, so I've stuck to the most trusted ones.

---

## Part 1 — Finance fundamentals (the part that actually matters)

### 1.1 What is an investment portfolio?

A **portfolio** is the collection of things you own that are meant to grow
your money over time. If you have $100 in a savings account, $500 in stocks,
and $200 in bonds, that's a $800 portfolio.

The job of a portfolio is to convert **today's money** into **more future
money**. The catch is that "more" isn't guaranteed — sometimes you get less.
Almost every concept that follows is about quantifying and managing that
uncertainty.

### 1.2 Asset classes — the building blocks

Different things you can buy behave very differently. Grouping them into
**asset classes** lets us reason about them at a higher level.

#### Stocks (equities)

When you buy a stock, you own a tiny slice of a company. If the company
makes money, the stock tends to go up. If the company loses money or the
economy hurts it, the stock goes down.

- **Individual stocks** (e.g. `AAPL`, `MSFT`, `NVDA`) — single companies.
  High potential return, but also high single-name risk: if Apple invents
  the next iPhone, AAPL skyrockets; if Apple has a scandal, AAPL crashes.
- **Stock index ETFs** (e.g. `SPY`) — bundle hundreds of stocks together.
  `SPY` tracks the S&P 500: the 500 largest US companies. If you buy
  `SPY`, you own a tiny piece of all 500. One company doing badly barely
  moves the needle.

This project uses both: individual names (`AAPL`, `MSFT`, `NVDA`) for the
"AI tilt" sleeve, and `SPY` as a passive index holding.

#### Bonds (fixed income)

A bond is a **loan** you make to a government or a company. You get fixed
interest payments, and at the end you get your money back. Bonds are
historically less volatile than stocks: they pay less but they don't crash
as hard.

The project uses `TLT` — an ETF that holds long-term US Treasury bonds (loans
to the US government, 20+ year maturity). Long-term Treasuries tend to **rise
when stocks crash** because investors flee to safety. That negative
correlation makes them a great diversifier (more on diversification below).

#### Commodities

Physical things: gold, silver, oil, wheat. The project uses `GLD`, an ETF
that effectively holds gold. Gold is famous for two things: keeping up with
inflation over very long periods, and going up when investors are panicking
about everything else.

#### International equities

US stocks are not the entire world. `EFA` is an ETF tracking large companies
in developed markets *outside* the US (Europe, Japan, Australia, etc.). When
the US market underperforms, international markets sometimes do better, and
vice versa.

**Why hold all five classes?** Because they don't all move together. We'll
get to that under "diversification" in a moment.

📚 *Learn more*:
- Wikipedia: [Asset class](https://en.wikipedia.org/wiki/Asset_classes)
- Investopedia: search for "asset class" — they have a clear explainer
- Book: *A Random Walk Down Wall Street* by Burton Malkiel (an institution
  in this space, written for non-experts)

### 1.3 Risk vs return — the only tradeoff that matters

This is the single most important concept in investing.

**Higher expected return almost always comes with higher risk.** That's not
a coincidence — if a low-risk investment offered high returns, everyone
would buy it, the price would shoot up, and the future return would drop.
Markets are pretty efficient at killing free lunches.

Concretely:

| Asset                | Long-term real return | Volatility |
|----------------------|----------------------|------------|
| Cash / savings       | ~0% above inflation  | Almost zero|
| Government bonds     | ~1–3%                | Low        |
| Diversified stocks   | ~5–7%                | Medium-high|
| Single-stock bets    | -100% to +∞          | Very high  |

When you build a portfolio, you're picking a **point on the risk/return
curve**. Conservative is "less return, less risk." Aggressive is "more
return, more risk." Balanced is in the middle.

This project's three profiles literally correspond to three points on that
curve:

```
Risk →
Conservative ◯——————————◯ Balanced ◯——————————◯ Aggressive
Lower return                                 Higher return
Lower volatility                             Higher volatility
```

📚 *Learn more*:
- Wikipedia: [Modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
  — the math behind risk/return tradeoffs. Don't try to grok the
  derivations on a first read; the *concept* is what matters.

### 1.4 Diversification — don't put all your eggs in one basket

If you put all your money in `AAPL` and Apple has a bad decade, you're
ruined. If you put all your money in stocks generally, and 2008 happens,
you're hurt but not ruined.

**Diversification** is the technical word for this. It rests on a single
mathematical idea: when assets don't move in perfect lockstep, holding
several of them reduces the *combined* volatility below what any single one
has on its own.

#### Correlation — the key term

Two assets are **positively correlated** if they tend to go up and down
together. **Negatively correlated** if when one goes up the other goes
down. **Uncorrelated** if they move independently.

| Asset pair             | Typical correlation |
|------------------------|---------------------|
| `AAPL` and `MSFT`      | High positive (~+0.6) — both big US tech |
| `SPY` and `EFA`        | Medium positive (~+0.4) — both stocks but different geographies |
| `SPY` (stocks) and `TLT` (bonds) | Low / mildly negative — when stocks fall, bonds often rise |
| `GLD` (gold) and stocks | Roughly zero — gold marches to its own drum |

The reason this project's profiles include bonds, gold, and international
exposure is **not** because we expect them to outperform US stocks — they
won't, in most decades. It's because they don't move with US stocks, so
holding some of them **smooths the ride**.

When stocks drop 20% and bonds gain 5%, a 60/40 stock/bond portfolio drops
about 10% instead of 20%. That smaller loss is the entire point.

📚 *Learn more*:
- Wikipedia: [Diversification (finance)](https://en.wikipedia.org/wiki/Diversification_(finance))
- Investopedia: search "correlation" and "diversification"
- Book: *The Bogleheads' Guide to Investing* (the bogleheads philosophy is
  built around diversification + low costs + discipline)

### 1.5 Volatility — measuring the rollercoaster

**Volatility** is the standard deviation of an asset's returns. In plain
English: how much do its daily returns wiggle around?

Suppose two stocks both return 10% per year on average:

- Stock A: +0.04% per day, every day, smoothly. Volatility ≈ 0.
- Stock B: +1% one day, -1% the next, +3% the day after, -2%, and so on.
  Volatility is high.

Both end up at the same place after a year, but **the journey is
completely different**. Stock B is gut-wrenching to hold. Volatility is
how we quantify that gut-wrench.

In this project, you'll see volatility computed in `src/features.py` as
`volatility_21d` (rolling 21-day standard deviation of daily returns).
That's a 1-month window. We use it as a feature for the ML models.

Annualized volatility is roughly the daily volatility times √252 (because
there are ~252 trading days in a year). A typical US stock has annualized
volatility around 15–25%. `TLT` (bonds) is more like 12–18%. `GLD` is
around 14%.

📚 *Learn more*:
- Wikipedia: [Volatility (finance)](https://en.wikipedia.org/wiki/Volatility_(finance))

### 1.6 Drawdown — measuring the worst pain

Volatility is the average wiggle. **Drawdown** is the worst pain.

Maximum drawdown is the biggest peak-to-trough drop your portfolio
experienced over a period.

If your portfolio went $1000 → $1500 → $1200 → $1800, the max drawdown
was the 1500→1200 drop = **-20%**, even though you ended up higher than
where you started.

Why does this matter? Because drawdown is what makes people panic and sell
at the bottom. A portfolio that returns 10%/year with a max drawdown of
-20% is psychologically much easier to hold than one that returns 12%/year
with a max drawdown of -50%. The first one keeps you in your seat. The
second one makes you sell at the worst moment and lock in losses.

In this project, the comparison table on the dashboard makes the tradeoff
explicit:

| Profile      | Total return | Max drawdown |
|--------------|-------------:|-------------:|
| Conservative | 27.77%       | -10.37%      |
| Balanced     | 59.65%       | -16.37%      |
| Aggressive   | 85.13%       | -20.87%      |

Aggressive earned 3× more, but its worst stretch was 2× as deep.

📚 *Learn more*:
- Wikipedia: [Drawdown (economics)](https://en.wikipedia.org/wiki/Drawdown_(economics))

### 1.7 Sharpe ratio — risk-adjusted return

Plain returns don't tell you whether you're being well-paid for the risk you
took. You can earn 50% by gambling everything on one stock — that's not
skill, that's luck. The **Sharpe ratio** tries to disentangle return from
risk.

The formula:

```
Sharpe = (annualized return − risk-free rate) / annualized volatility
```

In English: how much extra return did you earn per unit of volatility you
endured? Bigger is better.

Rough quality bands you'll see in finance:

| Sharpe        | Quality |
|---------------|---------|
| < 0           | You lost to T-bills. Bad. |
| 0 – 0.5       | Below average |
| 0.5 – 1       | Average |
| 1 – 2         | Decent. Most well-managed funds aim here. |
| 2 – 3         | Excellent |
| > 3           | Suspicious — recheck for bugs in the calculation |

In this project, the Balanced profile shows a backtest Sharpe of ~1.58
over 2024-2025. That looks great. Reality check: the test window was a
two-year bull market with relatively few stress events. **In a typical
2-year sample including a real crash, Sharpes drop hard.**

📚 *Learn more*:
- Wikipedia: [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
- Investopedia: search "sharpe ratio"

### 1.8 Sortino ratio — penalizing only downside

Sharpe treats *all* volatility as bad. But intuitively, an asset that has
*upside* surprises is fine — we only really care about *downside*
volatility (sudden losses).

The **Sortino ratio** uses the same formula as Sharpe but only counts the
volatility of *negative* returns. It's usually higher than the Sharpe and
arguably more honest, but Sharpe is the convention everyone reports.

📚 *Learn more*:
- Wikipedia: [Sortino ratio](https://en.wikipedia.org/wiki/Sortino_ratio)

### 1.9 Annualized return — comparing apples to apples

If you tell me a portfolio gained 60% over two years, that's not directly
comparable to one that gained 15% in 6 months. To make returns comparable,
we **annualize** them — express what the equivalent steady annual return
would have been.

The formula:

```
annualized = (1 + total_return) ^ (1 / years) − 1
```

Example from this project: Balanced earned 59.65% over ~2 years.
Annualized: (1.5965)^(1/2) − 1 ≈ **26.4%**.

Annualizing also stops people from cherry-picking time windows (you can't
say "I made 200% over 12 years" without admitting that's only ~9.7%/year).

### 1.10 Buy-and-hold vs. active strategies

Two opposing philosophies:

**Buy-and-hold (passive)**: pick a low-cost, broadly diversified set of
ETFs (e.g. just buy `SPY`), hold them forever, ignore the news. Studies
consistently show this beats most active managers, mostly because of
**costs**: trading isn't free, and active managers also charge fees.

**Active**: try to beat the market by picking winners, timing entries and
exits, or following a model's signals. Sometimes this works in any single
year. Over 10–20 years, the vast majority of active strategies
underperform a simple index after fees and taxes.

The honest framing of this project: **we are an active strategy that tries
to add a little bit of value via cross-asset rebalancing and a small AI
tilt, but the dominant story is the diversification itself, not the active
edge.** That's why the dashboard always shows your portfolio against a
"SPY buy-and-hold" benchmark — you should always check whether the
complicated thing actually beat the boring thing.

📚 *Learn more*:
- The Bogleheads forum (https://www.bogleheads.org/forum/) is the home of
  the buy-and-hold philosophy. Their wiki is excellent and free.
- Book: *Common Sense on Mutual Funds* by John Bogle (founder of Vanguard,
  pioneer of the index fund — strongly opinionated, very influential).
- Book: *A Random Walk Down Wall Street* (already mentioned).

### 1.11 Rebalancing — keeping the mix on target

You start with 60% stocks and 40% bonds. A year goes by, stocks did great,
bonds were flat. Now you have 70% stocks and 30% bonds. Your portfolio is
now **riskier than you wanted**.

**Rebalancing** means selling some of what's gotten too big and buying
more of what's shrunk, until you're back at 60/40. It does two things:

1. **Locks in your risk profile**. You stay at the level of risk you
   actually signed up for.
2. **Forces "buy low, sell high"**. The thing that's underweight is the
   thing that's recently underperformed; the thing that's overweight is
   what's recently surged. Rebalancing systematically trims winners and
   buys losers, which historically adds a small "rebalancing premium."

#### How often?

There's no single right answer, but the conventional wisdom:

- **Daily / weekly**: too often. Transaction costs and small movements
  drown the signal.
- **Monthly**: fine, but a bit anxious. Lots of noise.
- **Quarterly**: a sweet spot. Big enough drift to matter, low enough
  trading to stay cheap. ✅ ← *this project uses quarterly*
- **Annually**: also fine, especially for tax efficiency. But you can
  drift further from your target risk between rebalances.

Why specifically *quarterly* in this project? Because:
1. The ML model's signal is short-term (5–10 day horizon). Refreshing
   four times a year matches the rhythm of the underlying data without
   over-fitting.
2. It produces about 8 rebalances over 2024–2025 — enough to be
   meaningful, few enough to inspect by hand, which is what the
   "Decisions" tab in the app shows.
3. It limits transaction costs to a controllable amount.

📚 *Learn more*:
- Wikipedia: [Rebalancing investments](https://en.wikipedia.org/wiki/Rebalancing_investments)
- Bogleheads wiki: search "rebalancing"

### 1.12 Transaction costs — the silent killer

Every time you buy or sell, there's a cost. On retail platforms in 2025
this is usually:

- **Commissions**: $0 in most US brokers nowadays.
- **Spreads**: the price you buy at is slightly higher than the price you
  sell at. That gap is the broker's cut.
- **Slippage**: when you place a market order, the price might move
  slightly against you between when you click and when you fill.
- **Taxes** (in real life): selling for a profit triggers capital gains
  tax. The simulator does NOT model this.

This project models a flat **10 basis points** (0.10%) per trade. So
buying $1000 of a stock costs $1.

That sounds tiny. But: if you trade often enough, even small costs erode
returns. A strategy that rebalances every 5 days with 10 bps cost loses
roughly 50 × 0.10% = 5% per year just to fees. That's the difference
between a great year and a mediocre year. **This is one of the strongest
reasons we rebalance quarterly, not weekly.**

📚 *Learn more*:
- Wikipedia: [Basis point](https://en.wikipedia.org/wiki/Basis_point)
  (1 bp = 0.01%)

### 1.13 Bull and bear markets

- **Bull market** = sustained period of rising prices. Optimism, FOMO,
  rising P/E ratios. (E.g. 2020-2021, 2024-2025.)
- **Bear market** = sustained period of falling prices, conventionally
  defined as -20% from peak. Panic, forced selling, capitulation. (E.g.
  2008-2009, March 2020.)

Strategy backtests done **only in bull markets** look fantastic. The same
strategies usually look much worse when extended through a bear market.
Always check what kind of market a backtest covers.

This project's test window (2024-01-02 → 2025-12-31) is a bull market.
Treat the headline returns with appropriate skepticism.

### 1.14 Ways to fool yourself with a backtest

Real talk: it's *very* easy to construct a backtest that looks brilliant
on paper but doesn't generalize. Things to watch for in any strategy
(including this one):

- **Survivorship bias**: only testing on tickers that still exist today.
  Companies that went bankrupt are missing from your dataset, making
  your strategy look better than reality.
- **Look-ahead bias**: accidentally letting the model peek at future
  data. This is *the* most common subtle backtest bug.
- **Overfitting**: trying enough strategies that one of them looks great
  by pure chance. Statistical pattern: the more knobs you turn, the more
  you'll get a fit on the past that won't repeat.
- **Selection bias**: picking the best-performing model variant from
  many, then only reporting that one. (We do this in this project but
  we use a separate validation period to mitigate it — more on this in
  Part 3.)
- **Idealized frictions**: zero slippage, instant fills, no taxes, no
  market impact from your trades. Real markets are messier.
- **Single-period testing**: a strategy that works in one decade may not
  work in another. The market regime changes (interest rates, tech
  cycles, geopolitics).

📚 *Learn more*:
- Wikipedia: [Backtesting](https://en.wikipedia.org/wiki/Backtesting)
- Book: *Advances in Financial Machine Learning* by Marcos López de Prado
  (Chapter 11 has a deep dive on backtest pitfalls; the book is dense
  but the warnings are top-tier).

---

## Part 2 — How this project's strategy actually works

Now that you know the building blocks, here's the project's strategy in
those terms.

### 2.1 The three risk profiles

Each profile is a **fixed cross-asset allocation that sums to 100%**. The
allocations are decided by the user once (during onboarding) and don't
change based on market conditions. The portfolio rebalances back to these
targets every quarter.

| Asset class           | Conservative | Balanced | Aggressive |
|-----------------------|-------------:|---------:|-----------:|
| US individual stocks  | 20%          | 45%      | 65%        |
| Long-term Treasuries  | 60%          | 30%      | 10%        |
| S&P 500 ETF           | 10%          | 15%      | 15%        |
| Gold                  | 5%           | 5%       | 5%         |
| International stocks  | 5%           | 5%       | 5%         |

Looking at this through the diversification lens:
- **Conservative** dominates with bonds (60%) — heavy ballast against
  stock crashes. Good for someone close to needing the money.
- **Balanced** is a textbook 60/40-ish stock/bond mix. The historic
  workhorse for medium-term goals.
- **Aggressive** is mostly stocks (80%+ if you count individual + index
  + international). Higher expected return, more drawdown risk.

The 5%/5%/5% slivers (S&P ETF, gold, international) are present in *all*
profiles to ensure baseline diversification even at the conservative end.

### 2.2 Why these specific allocations?

These are not from a quantitative optimizer; they're **conventional rules
of thumb** distilled from decades of personal-finance literature. The
60/40, 80/20, and 20/60 splits roughly map to industry-standard "moderate,"
"aggressive," and "conservative" portfolios.

You could optimize them with mean-variance optimization or risk parity (see
Part 8 resources for both). For a beginner-facing demo, simple round
numbers are more honest — they're not tuned to the test data, so they
won't suffer as much from overfitting.

### 2.3 The AI tilt

The "AI tilt" is the only place the ML model touches the portfolio. It
applies **inside the equity sleeve only** (the US individual stocks
slice).

Mechanism:
1. Equity sleeve starts equal-weighted across the names (e.g. Aggressive
   profile equity sleeve = 65%; with 3 names = 21.67% each).
2. The ML model produces a "score" for each name (predicted future
   return). Names are ranked.
3. Each name's weight is nudged by up to ±5% based on its score, with two
   constraints:
   - **Capped at ±5%** absolute. The strongest signal can move a name
     from 21.67% to 26.67% or 16.67%, no more.
   - **Zero-sum**. The over-weights and under-weights cancel exactly, so
     the equity sleeve total stays at 65% (or whatever the profile
     dictates). Cross-asset risk is unchanged.
4. The actual trades happen at the next quarterly rebalance.

Why these constraints?
- **The cap** ensures the AI can't blow up your risk profile if it's
  wrong. If the model develops a wild bias toward NVDA, the worst case is
  a small overweight, not 90% of the portfolio.
- **Zero-sum** ensures you always have the diversification benefit of the
  profile. Without zero-sum, a tilt that shrank one stock without growing
  another would silently shift money to cash, ruining the cross-asset
  exposure.
- **Inspectable**: every tilt produces a human-readable rationale shown
  in the app's "Decisions" tab. *Transparent AI is a feature, not a
  marketing line.*

#### Honest result

In the 2024-2025 backtest, the tilt added roughly +0.5% to +1% total
return per profile, but slightly *increased* max drawdown. **Risk-
adjusted (Sharpe), the tilt is essentially a wash.** That's why the app
defaults the tilt OFF and lets users opt in.

This is a feature of being honest, not a failure. The real story of this
strategy is the profile (diversification + discipline). The AI is a small
optional experiment on top.

### 2.4 The end-to-end flow on rebalance day

Walking through what happens on, say, 2024-04-01 (Q2 2024 rebalance):

1. **Wake up**. The simulator sees today is in the quarterly rebalance
   list.
2. **Compute new target weights**:
   - Look up the profile's targets (e.g. Balanced = 45/30/15/5/5).
   - For non-equity sleeves, just use the index-proxy ticker (`TLT` for
     bonds, `SPY` for index, `GLD` for gold, `EFA` for international)
     at the profile target weight.
   - For the equity sleeve, fetch the latest ML score for each ticker,
     compute the capped zero-sum tilts, and produce per-ticker target
     weights summing to the equity sleeve target.
3. **Compute trades**:
   - For each ticker, compare current weight (where the portfolio drifted
     to since the last rebalance) to target weight.
   - If different, buy or sell the difference. Fractional shares are
     allowed, so the exact target is achievable.
4. **Apply transaction costs**: 10 basis points of every trade.
5. **Generate a rationale**: a sentence-by-sentence English description
   of the new allocation, the tilts (if any), the reasoning. This is
   what shows in the app under "Decisions → tap a date."
6. **Record**: the day's trades, post-rebalance portfolio value, and the
   target weights are all saved for the daily log.

Every day in between rebalances, the simulator just marks positions to
the day's closing prices and records the new portfolio value. No trades
happen. The portfolio drifts away from target weights as prices move,
and the next quarterly rebalance pulls it back.

---

## Part 3 — Backtesting and being honest

### 3.1 What is a backtest?

A **backtest** is "what would have happened if I'd run this strategy in
the past." You take historical price data, simulate the strategy day by
day, and report what the portfolio looked like at the end.

Backtests are useful for:
- Catching obvious bugs (does the strategy even work mechanically?)
- Comparing strategies against each other and against benchmarks
- Building intuition about how a strategy behaves

Backtests are NOT useful for:
- Predicting future returns. Past behavior is correlated with but does
  not equal future behavior.
- Validating you'll outperform the market. The probability of fooling
  yourself is high (see 1.14 above).

### 3.2 Train / val / test splits

When training an ML model on price data, you split your data into three
chronological pieces:

```
2015 ─── 2021 ─── 2023 ─── 2025
└─ train ─┘ └─ val ─┘ └─ test ─┘
```

- **Train** (2015–2021): the model learns patterns from this data.
- **Validation** (2022–2023): we evaluate the model here to *pick* which
  one is best (regression vs classification, which target, etc.).
- **Test** (2024–2025): the model has *never seen this data* during
  training or selection. We report numbers from here as "out-of-sample."

The chronological split matters: you can't randomly shuffle financial
data because that lets the model peek at future patterns when training
on past ones. The future leaks into the past — look-ahead bias.

📚 *Learn more*:
- scikit-learn docs on [time-series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- *Advances in Financial Machine Learning*, López de Prado — chapters on
  cross-validation and proper backtesting.

### 3.3 Selection bias (and how we mitigate it here)

Imagine you train 100 models and report the one with the best test set
performance. *Of course* one of them looks great by chance — out of 100
random walks, one will look like a strategy.

In this project we try ~20 (model, target) combinations. To avoid lying
to ourselves:
1. **Selection** is done on the **validation** set, not the test set.
2. The **test set is reported only after** selection is locked in.

This is the textbook approach. It's still imperfect — the training and
validation periods inform our choices implicitly — but it's much more
honest than reporting on test-set selection.

### 3.4 Why the backtest doesn't predict the future

The 2024-2025 test window has these biases baked in:
- **Bull market**. Most things went up. Our drawdowns are mild compared
  to what 2008 or 2022 looked like.
- **Mostly-known regime**. The market structure (Fed policy, interest
  rates, sector composition) is broadly similar to recent years. A
  strategy tuned implicitly to recent regimes won't necessarily do well
  if rates collapse or AI hype bursts.
- **Three stocks**. The equity sleeve is `AAPL`, `MSFT`, `NVDA`. All
  three were among the strongest performers in the entire market. A
  random sample of 3 stocks would have looked very different.

Treat the backtest as a **plausibility check** — does the strategy do
something sensible? — not as a forecast.

---

## Part 4 — The ML side, lightly

This section is intentionally shallower than the finance content. The
goal is to give you enough to follow the code.

### 4.1 Supervised learning in 90 seconds

Supervised learning is teaching a model to map **inputs (features)** to
**outputs (targets)** by showing it lots of examples with the answer.

For our use case:
- **Inputs**: today's market data — what's the volatility, what's the
  RSI, where is price relative to the 50-day moving average, etc.
- **Outputs**: future return — what will this stock do over the next
  5 days? 10 days?
- **Training**: show the model thousands of examples (date, features,
  realized future return) and let it learn the relationship.
- **Inference**: given a fresh set of features, produce a prediction.

📚 *Learn more*:
- Andrew Ng's [Machine Learning Specialization on Coursera](https://www.coursera.org/specializations/machine-learning-introduction)
  — the canonical beginner course.
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html

### 4.2 Features in this project

`src/features.py` engineers ~54 features per ticker per day. Categories:

- **Returns over various windows**: 1d, 5d, 10d, 21d, 63d returns.
- **Moving averages**: 10, 20, 50, 200-day moving averages and the gap
  between them and current price (a momentum-ish signal).
- **Volatility**: rolling std of returns, separated into total and
  downside.
- **RSI**: a classic technical indicator measuring "is this stock
  overbought or oversold relative to recent action?"
- **MACD**: another classic — difference between fast and slow EMAs.
- **Volume features**: relative volume vs recent average — high volume
  often signals significant news.
- **Market context**: the same features computed for the broader market
  (`^GSPC`), and the stock's beta (correlation × relative volatility) to
  the market.
- **Cross-sectional features**: how does this ticker rank vs the others
  on the same day?

These are not magic — they're standard technical analysis features that
discretionary traders have used for decades. The ML model finds patterns
in their *combinations*.

### 4.3 Targets

We try several targets to predict:
- `target_5d_return` — the realized return 5 trading days ahead.
- `target_10d_return` — same but 10 days ahead.
- `target_5d_risk_adj_return` — return divided by recent volatility.
- `target_5d_direction` — binary: did the next 5-day return end up
  positive or negative?

Different targets train different kinds of models (regression for
continuous returns, classification for direction). We try several to see
which actually has a usable signal.

### 4.4 The seven models we try

The project trains seven models from scikit-learn:

- **Ridge regression** — linear model with L2 regularization. Simple,
  fast baseline.
- **ElasticNet regression** — linear with both L1 and L2 regularization.
  Slightly more flexible than Ridge.
- **Histogram Gradient Boosting (regression)** — modern ensemble of
  decision trees. Often strong on tabular data.
- **Random Forest (regression)** — another tree ensemble; bagging
  instead of boosting.
- **Logistic regression (classification)** — linear classifier for the
  direction targets.
- **Histogram Gradient Boosting (classification)** — the classifier
  version of GBT.
- **Random Forest (classification)** — classifier version of RF.

Why try all of them? Because no single model wins on every dataset, and
trying a portfolio of them — then picking the best on the validation
set — is a robust default.

📚 *Learn more*:
- scikit-learn user guide chapters on each model (Ridge, ElasticNet,
  GBT, RF, Logistic — all linked from the user guide above).
- Book: *Hands-On Machine Learning with Scikit-Learn, Keras, and
  TensorFlow* by Aurélien Géron — the most accessible "build it
  yourself" intro to applied ML I know of.

### 4.5 What "validation Sharpe" actually means

For each (model, target) combination we don't just look at how well it
predicted returns — we go a step further:

1. Use the model's predictions to rank stocks each day.
2. Construct a hypothetical portfolio that buys the top-ranked names.
3. Compute the **Sharpe ratio** of that portfolio over the validation
   period.
4. Pick the (model, target) with the best Sharpe.

This is more honest than picking by raw R² or accuracy because it ties
the model directly to *trading economics*. A model with okay R² but
profitable trades beats one with great R² that always picks losers.

This whole process is in `main.py` — search for `val_backtest_sharpe`.

---

## Part 5 — How the code maps to all this

A bird's-eye view so you can navigate.

### The Python side (`risk-return-analysis/`)

| File                     | What it does in plain English |
|--------------------------|-------------------------------|
| `src/data_fetch.py`      | Downloads historical prices from Yahoo Finance via `yfinance`. |
| `src/features.py`        | Computes the ~54 features per ticker per day (RSI, MACD, moving averages, etc.). |
| `src/model.py`           | Trains and evaluates the seven ML models on multiple targets. |
| `src/profiles.py`        | Defines Conservative/Balanced/Aggressive and their target weights. |
| `src/profile_strategy.py`| The brain: quarterly rebalance dates, capped zero-sum AI tilt math, rationale generation. |
| `src/simulation.py`      | The dollar-based simulator: takes a target-weight timeline, walks day by day, records trades and portfolio value. |
| `src/export_app_data.py` | Writes the JSON files the app reads. |
| `main.py`                | The orchestrator: fetch → features → train → simulate × 6 → export. |

### The bridge

```
risk-return-analysis/main.py
   ↓ writes JSON to ../app/assets/data/
   ↓ summary.json + 6 detail files
app/data/loadData.ts
   ↓ static `import` (Metro bundles at build time)
React Native components
```

No backend. No live model. No network. The app is purely a presentation
layer for the precomputed simulation results, scaled linearly to the
user's chosen capital.

### The Expo side (`app/`)

| File / dir                       | What it does |
|----------------------------------|--------------|
| `app/onboarding.tsx`             | Profile picker + capital + tilt toggle on first launch. |
| `app/(tabs)/index.tsx`           | Dashboard: portfolio value chart vs SPY, key metrics, current allocation bar. |
| `app/(tabs)/timeline.tsx`        | List of all 8 quarterly rebalance events. |
| `app/(tabs)/about.tsx`           | What this is, ML model info, switch profile/tilt. |
| `app/event/[date].tsx`           | Tap a date → rationale, allocation, AI tilts, trades. |
| `state/AppContext.tsx`           | Persists profile/capital/tilt to AsyncStorage. |
| `lib/format.ts`                  | All currency/percent/date formatting; the `scaleFromBase($1k → user capital)` helper. |
| `components/PortfolioChart.tsx`  | The line chart with benchmark overlay. |
| `components/AllocationBar.tsx`   | The horizontal asset-class breakdown. |

---

## Part 6 — Honest limitations of this project

A non-exhaustive list of things you should **not** infer from this
project:

1. **It's not investment advice.** A simulation cannot model your
   personal financial situation, taxes, time horizon, or psychological
   tolerance for losses.

2. **The 3-stock equity sleeve is too narrow** to be a real product.
   `AAPL`, `MSFT`, `NVDA` were the strongest mega-caps in the test
   window. Any strategy that overweights them looks brilliant during
   2024-2025 and might look terrible in a different period. Expanding to
   ~8–15 names (and refitting the ML) would be a much better product.

3. **The test window is one bull market.** Results from one regime
   rarely transfer to other regimes.

4. **No taxes, no slippage beyond 10 bps.** Real frictions are bigger.

5. **No dividends.** `yfinance` returns adjusted-close prices that
   include dividends, so total return is approximated, but
   reinvestment-timing effects are not modeled exactly.

6. **No regime detection.** The strategy doesn't know what kind of
   market it's in. A real product would probably reduce equity exposure
   when volatility spikes (a "vol target") or pull back during
   recessions.

7. **The AI tilt is essentially neutral on Sharpe.** Don't oversell it.

8. **Past performance does not predict future results.** You will see
   this disclaimer everywhere in finance because it's a regulatory
   requirement *and* it's true.

A real product built around this skeleton would also need:
- Compliance review (regulators care a lot about anything that smells
  like advice).
- A way to enter and exit specific market regimes (e.g. raise cash in
  bear markets).
- Tax-aware rebalancing (avoid generating short-term capital gains).
- A larger universe with regular re-selection of holdings.
- Live data feeds + actual brokerage integration.

---

## Part 7 — Glossary (skim when you need it)

| Term                  | Definition |
|-----------------------|------------|
| Alpha                 | Excess return above a benchmark; the "skill" component of a strategy. |
| Annualized return     | Average yearly compounding rate, used to compare returns across windows. |
| Asset class           | A category of investment with shared characteristics (stocks, bonds, etc.). |
| Backtest              | Simulating a strategy on historical data. |
| Basis point (bp)      | 1/100 of 1%. 100 bps = 1%. |
| Bear market           | Sustained price decline (typically -20% from peak). |
| Beta                  | An asset's sensitivity to market moves; ~1.0 = moves with the market. |
| Bull market           | Sustained price rise. |
| Correlation           | Statistical measure of how two assets move together (-1 to +1). |
| Diversification       | Combining uncorrelated assets to reduce overall risk. |
| Drawdown              | Decline from a recent peak. |
| ETF                   | Exchange-Traded Fund; a basket of assets traded like a stock. |
| Equity                | A share in a company; "stocks." |
| Feature               | An input variable the ML model uses to predict. |
| Fixed income          | Bond-like assets that pay predictable cash flows. |
| Fractional share      | Owning less than 1 share, possible at modern brokers. |
| Index                 | A measurement basket (S&P 500, Nasdaq, FTSE, etc.). |
| Look-ahead bias       | A backtest bug where the model uses future data to predict the present. |
| Market order          | Buy/sell at the current best available price. |
| Max drawdown          | The deepest peak-to-trough decline observed. |
| Momentum              | The tendency for recent winners to keep winning short-term. |
| Out-of-sample         | Data the model has never seen; "test set." |
| Overfitting           | A model that fits the training data too closely and fails on new data. |
| Position              | An open holding in a specific asset. |
| Portfolio             | The collection of positions and cash. |
| Rebalancing           | Trading back to the target weights. |
| Risk-free rate        | The return on the safest available asset (usually short Treasury bills). |
| Sharpe ratio          | Excess return per unit of volatility. Higher is better. |
| Slippage              | Difference between expected price and fill price. |
| Sortino ratio         | Like Sharpe but only counts downside volatility. |
| Spread                | Difference between bid (sell) and ask (buy) prices. |
| Survivorship bias     | A dataset bug where dead/delisted tickers are missing. |
| Target weight         | The intended allocation to a position. |
| Test set              | Data used only to evaluate the final chosen model. |
| Tilt (in this project)| A capped, zero-sum adjustment to per-ticker weights inside the equity sleeve. |
| Train set             | Data used to fit the model. |
| Transaction cost      | The cost of placing a trade (commissions, spreads, slippage). |
| Validation set        | Data used to choose between candidate models. |
| Volatility            | Standard deviation of returns; measures the size of price swings. |
| Yield                 | Income (interest, dividends) divided by price. |

---

## Part 8 — Curated learning resources

I've cut a lot of options. The world is full of confident people selling
financial education that's worse than useless. The list below is
deliberately conservative.

### 8.1 Books (in priority order)

1. **A Random Walk Down Wall Street** — Burton Malkiel. The canonical
   "passive investing for the educated public" book. If you read one
   finance book in your life, make it this one. Multiple editions over
   ~50 years.
2. **The Bogleheads' Guide to Investing** — multiple authors. The
   practical companion: how to actually do passive investing in real
   life. Less philosophical, more "open this account, buy that fund."
3. **Common Sense on Mutual Funds** — John Bogle. The founder of
   Vanguard, with strong opinions on fees, indexing, and the financial
   industry.
4. **Hands-On Machine Learning with Scikit-Learn, Keras, and
   TensorFlow** — Aurélien Géron. The most approachable applied-ML
   intro. Read this if Andrew Ng's course feels too math-heavy.
5. **Advances in Financial Machine Learning** — Marcos López de Prado.
   *Advanced.* Don't start here. Read it once you've done some real ML
   work. Best resource on backtest pitfalls and time-series ML done
   properly.

### 8.2 Online courses

- [Coursera — Andrew Ng: Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction).
  The bedrock ML course. Math-heavy in spots but every minute is worth it.
- [Khan Academy — Finance and Capital Markets](https://www.khanacademy.org/economics-finance-domain/core-finance).
  Free, well-paced finance basics. Start here if "stocks" and "bonds"
  feel hazy.
- [Coursera — Investment Management Specialization (Geneva)](https://www.coursera.org/specializations/investment-management)
  for a more academic take on portfolio construction.

### 8.3 Reference sites

- [Investopedia](https://www.investopedia.com/) — search-engine-quality
  definitions of every term in finance. The "Terms" pages are usually
  trustworthy; the "Articles" pages are sometimes ad-driven, so prefer
  the term pages.
- [Wikipedia](https://en.wikipedia.org/) — for the math / theory side
  of any topic. The articles for major concepts like
  [Modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory),
  [Sharpe ratio](https://en.wikipedia.org/wiki/Sharpe_ratio),
  [Diversification (finance)](https://en.wikipedia.org/wiki/Diversification_(finance))
  are excellent.
- [Bogleheads wiki](https://www.bogleheads.org/wiki/Main_Page) — the
  community wiki for the index-investing crowd. Practical, well-edited,
  free.

### 8.4 Communities

- [r/Bogleheads](https://reddit.com/r/Bogleheads) — passive investing
  with sane defaults and minimal noise.
- [r/personalfinance](https://reddit.com/r/personalfinance) — life-stage
  questions (paying off debt, building emergency funds, retirement
  accounts).
- [Bogleheads forum](https://www.bogleheads.org/forum/) — slower-moving,
  higher-signal version of the subreddit.

> ⚠️ **Avoid**: r/wallstreetbets, "fintwit" gurus, anyone selling a
> course or trading signals. The ratio of bad-to-good content in those
> spaces is severe.

### 8.5 Project-specific docs

- [yfinance on PyPI](https://pypi.org/project/yfinance/) — the library
  that fetches historical prices in `data_fetch.py`.
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
  — every model the project uses is documented here.
- [pandas docs](https://pandas.pydata.org/docs/) — the data structures
  you'll see everywhere in the Python pipeline.
- [Expo documentation](https://docs.expo.dev/) — the React Native
  framework powering the app.
- [React Native docs](https://reactnative.dev/) — for the UI patterns.

---

## A study path I'd suggest

If you want to actually internalize the financial parts of this project,
the fastest path I'd suggest:

1. Read Khan Academy's "Stocks and bonds" sections (~2 hours).
2. Read Investopedia entries for: portfolio, asset class, diversification,
   correlation, volatility, Sharpe ratio, drawdown, rebalancing, ETF,
   bond, basis point. (~1 evening.)
3. Read **A Random Walk Down Wall Street** front to back (~2 weeks at a
   reasonable pace).
4. Come back to this project's code and read `main.py` end-to-end.
   You'll find that almost every line maps onto a concept you now know.
5. Read **The Bogleheads' Guide to Investing** for the practical side
   (~1 week).
6. (Optional, much later) When you want to build something more
   advanced: Géron's ML book → López de Prado.

---

## Final word

This project is intentionally simple, transparent, and honest. The
machinery looks technical, but the underlying ideas are mostly
century-old wisdom: diversify, rebalance with discipline, keep costs
low, don't let one bad event bankrupt you. The ML on top is a small
optional enhancement, transparently shown in the rationale text on
every rebalance.

If a section here doesn't click on first read, that's normal — finance
is one of those fields where concepts only really land after you've seen
them three or four times in different contexts. Come back to this doc
periodically as you read elsewhere.
