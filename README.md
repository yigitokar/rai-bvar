# rai-bvar

Ablation-ready Bayesian VARs for Python. Models are compositions of a **mean**, a **volatility** and a **prior** component; one backtest engine and one predictive contract evaluate them all, and a verdict engine decides whether extra complexity actually improves out-of-sample density forecasts.

> **Status: Phase 0 (foundations).** The data layer, predictive contract, scores, significance tests, backtest/compare engine, benchmark models (random walk, OLS VAR) and dataset loaders are in place. Bayesian models arrive in Phase 1 (conjugate Minnesota) and Phase 2 (MCMC with stochastic volatility). See [SPEC.md](SPEC.md).

## Quick start

```python
import rai_bvar as rb

data = rb.datasets.load("fred-qd", vintage="2026-09", variables="small")   # downloads once, verifies checksum

res = rb.compare(
    {"RW": rb.RandomWalk(), "OLS-VAR(2)": rb.OLSVAR(lags=2)},
    data,
    origins=rb.Origins(start="2000Q1"),
    window=rb.Rolling(size=120),
    horizons=[1, 4],
    baseline="RW",
)
res.table()
res.verdict()
```

## Development

```bash
uv sync --group dev
uv run pytest                      # fast suite (what CI runs on every push)
uv run pytest -m "slow or network" # nightly suite
uv run ruff check . && uv run ruff format --check .
```

Licensed under Apache-2.0.
