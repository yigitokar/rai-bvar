# rai-bvar

Ablation-ready Bayesian VARs for Python. Models are compositions of a **mean**, a **volatility** and a **prior** component; one backtest engine and one predictive contract evaluate them all, and a verdict engine decides whether extra complexity actually improves out-of-sample density forecasts.

> **Status: Phase 2 (analytic and MCMC engines).** The analytic engine (Phase 1) is the conjugate Minnesota BVAR: SOC/SUR dummies, the GLP (2015) hierarchy (empirical Bayes or Metropolis-Hastings over hyperparameters), Lenza-Primiceri COVID scaling and recursive IRFs/FEVD, at numerical parity with R `BVAR` and the Bank of England `bvar`. The MCMC engine (Phase 2) is a hand-written JAX Gibbs sampler for the `Linear` mean in structural or reduced form (the corrected CCCM 2022 sampler) × `Constant`, `SV`, `SV(rw=True)`, `SV(outliers=True)` and `SV(tails="t")`, with SOC/SUR dummies, empirical-Bayes Minnesota hyperparameters, runtime convergence diagnostics, an ordering-sensitivity check and an opt-in path-integrated log score (`lpd_method="integrated"`). Every composition has Geweke and SBC tests (some SBC rows are to be re-run after later kernel changes), and the engine is checked against R `stochvol` and, for the reduced form, an exact system-wide Gibbs oracle. Exit criteria, evidence and open items: [SPEC.md §13](SPEC.md#phase-2-mcmc-engine-and-the-volatility-axis); design: [docs/phase2/DESIGN.md](docs/phase2/DESIGN.md). Next: Phase 3 (global-local shrinkage, `CommonSV`, conditional forecasts).

## Quick start

```python
import rai_bvar as rb

data = rb.datasets.load("fred-qd", vintage="2026-09", variables="small")   # downloads once, verifies checksum

# Analytic engine: conjugate Minnesota prior with empirical-Bayes hyperparameters (seconds)
bvar = rb.BVAR(
    mean=rb.Linear(prior=rb.Minnesota(soc=1.0, sur=1.0, hierarchy="eb")),
    lags=4,
    outliers="lenza-primiceri",                # the sample covers 2020 (SPEC §5.6)
)
fit = bvar.fit(data)
fit.summary()                                  # hyperparameter mode (λ, μ, δ)
fit.forecast(horizon=8, draws=2000, seed=0).quantiles([0.05, 0.5, 0.95])
fit.irf(horizon=20, draws=1000, seed=0).median()

# MCMC engine: stochastic volatility with an outlier component (Minnesota() resolves to the
# structural form; 4 chains × 2,000 draws by default)
sv_fit = rb.BVAR(vol=rb.SV(outliers=True), lags=4).fit(data)
sv_fit.converged                               # R-hat / ESS gate (DESIGN §8.2): False here (see below)
sv_fit.diagnostics                             # R-hat, ESS and MCSE per monitored quantity
sv_fit.forecast(horizon=8, draws=2000, seed=0).quantiles([0.05, 0.5, 0.95])

# Does stochastic volatility improve the density forecasts? The two models differ only in their
# volatility, so the verdict engine can decide (about 3.5 minutes on an Apple M5 Max)
mcmc = rb.MCMCSettings(draws=20_000)           # enough draws for the convergence gate (below)
res = rb.compare(
    {
        "Constant": rb.BVAR(vol=rb.Constant("ig"), lags=4, mcmc=mcmc),
        "SV(outliers)": rb.BVAR(vol=rb.SV(outliers=True), lags=4, mcmc=mcmc),
    },
    data,
    origins=rb.Origins(start="2016Q1", step=4),
    window=rb.Rolling(size=120),
    horizons=[1, 4],
)
res.table()
res.verdict()                                  # SV(outliers) vs Constant along "vol", per horizon
```

On the 2026-09 vintage `res.verdict()` returned `better` at both horizons (Apple M5 Max, 2026-10-23; MCMC draws are reproducible on one platform, not across platforms, SPEC §8). A verdict is one of `better`, `worse` or `inconclusive`, with its reasons. It is only issued between two models that differ along exactly one axis (SPEC §3 R2, §9.5): `Constant("ig")` keeps both arms on the MCMC engine with the same structural Minnesota prior, while `Constant()` would select the analytic conjugate model and change the prior too. Comparing, say, `rb.RandomWalk()` with a BVAR gives `not comparable`, because they differ in the mean, prior, volatility, lags and more. The verdict also requires converged fits at 95% of the origins or more. At the default 4 × 2,000 draws the SV fits of this example miss the gate (R-hat < 1.01 and bulk ESS > 400 on the primary monitors, mostly the SV σ and the log-likelihood) at every origin, as the full-sample `sv_fit` above does, and the verdict is then `inconclusive` with the convergence reason listed; the default convergence policy is open question Q14 (DESIGN §12.2).

## Development

```bash
uv sync --group dev
uv run pytest                                       # fast suite (what CI runs on every PR)
uv run pytest -m "(slow or network) and not bench"  # weekly suite (CI: Sundays): G-full, SBC, long parity, mutation
uv run pytest -m bench -s                           # pre-release BENCH-P2 and timing gates (DESIGN §10)
Rscript tools/make_r_fixtures.R --sections=bvar     # regenerate the R BVAR parity fixtures (needs R + BVAR)
uv run python -I tools/make_sv_parity_data.py all   # Phase 2: data, R stochvol and system-wide Gibbs runs, NetCDF
uv run ruff check . && uv run ruff format --check .
```

Licensed under Apache-2.0.
