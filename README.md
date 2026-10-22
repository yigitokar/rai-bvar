# rai-bvar

Ablation-ready Bayesian VARs for Python. Models are compositions of a **mean**, a **volatility** and a **prior** component; one backtest engine and one predictive contract evaluate them all, and a verdict engine decides whether extra complexity actually improves out-of-sample density forecasts.

> **Status: Phase 1 (analytic engine).** Phase 0 foundations plus the conjugate Minnesota BVAR: SOC/SUR dummies, the GLP (2015) hierarchy (empirical Bayes or Metropolis-Hastings over hyperparameters), Lenza-Primiceri COVID scaling, and recursive IRFs/FEVD, with numerical parity against R `BVAR` and the Bank of England `bvar`. MCMC models with stochastic volatility arrive in Phase 2. See [SPEC.md](SPEC.md).

## Quick start

```python
import rai_bvar as rb

data = rb.datasets.load("fred-qd", vintage="2026-09", variables="small")   # downloads once, verifies checksum

bvar = rb.BVAR(
    mean=rb.Linear(prior=rb.Minnesota(soc=1.0, sur=1.0, hierarchy="eb")),
    lags=4,
    outliers="lenza-primiceri",                # the sample covers 2020 (SPEC §5.6)
)
fit = bvar.fit(data)
fit.summary()                                  # hyperparameter mode (λ, μ, δ)
fit.forecast(horizon=8, draws=2000, seed=0).quantiles([0.05, 0.5, 0.95])
fit.irf(horizon=20, draws=1000, seed=0).median()

res = rb.compare(
    {"RW": rb.RandomWalk(), "OLS-VAR(2)": rb.OLSVAR(lags=2), "BVAR(4)": bvar},
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
Rscript tools/make_r_fixtures.R    # regenerate R BVAR parity fixtures (needs R + BVAR)
uv run ruff check . && uv run ruff format --check .
```

Licensed under Apache-2.0.
