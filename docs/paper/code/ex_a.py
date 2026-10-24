# Listing 1 of the paper (docs/paper/sections/07_walkthrough.tex), shown from line 6 on.
# Run from the main repository: cd /Users/yigitokar/rai-bvar && uv run python <this file>,
# or through run_examples.py. Output captured at main-repo commit 4ac7969 (release 0.2.0,
# clean tree; out/env.json); identical to the earlier capture at d58c9ef.

import numpy as np
import pandas as pd

import rai_bvar as rb

# A tiny quarterly data set: a stationary VAR(1) in (GDP growth, inflation, rate),
# stored as raw levels (two log-levels, one rate) with FRED-style transformation codes.
rng = np.random.default_rng(0)
A = np.array([[0.4, 0.0, -0.001], [0.1, 0.6, 0.0], [15.0, 25.0, 0.85]])
mu = np.array([0.0075, 0.006, 4.0])
sd = np.array([0.007, 0.003, 0.25])
corr = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.2], [0.3, 0.2, 1.0]])
L = np.linalg.cholesky(corr * np.outer(sd, sd))
y, Y = mu.copy(), []
for _ in range(200 + 240):
    y = (np.eye(3) - A) @ mu + A @ y + L @ rng.standard_normal(3)
    Y.append(y)
Y = np.array(Y[200:])
raw = pd.DataFrame(
    {"gdp": 100 * np.exp(np.cumsum(Y[:, 0])), "prices": 50 * np.exp(np.cumsum(Y[:, 1])),
     "rate": Y[:, 2]},
    index=pd.period_range("1960Q1", periods=240, freq="Q"),
)
data = rb.VARData.from_frame(raw, tcodes={"gdp": 5, "prices": 5, "rate": 1})
print(data)
print("prior means of own first lag:", data.prior_means())

# Analytic engine: conjugate Minnesota with SOC/SUR dummies and GLP empirical Bayes.
bvar = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(soc=1.0, sur=1.0)), lags=2)
print(bvar.engine, "|", bvar.name)
fit = bvar.fit(data)
print(fit.summary())
fc = fit.forecast(horizon=4, draws=2000, seed=0)
print(fc)
print(fc.quantiles([0.05, 0.5, 0.95]).sel(horizon=1).round(4).to_pandas())
print("paths_level h=1 median:", fc.median(level=True).sel(horizon=1).round(3).values)

# Pseudo out-of-sample comparison on rolling windows (valid DM/AG p-values).
res = rb.compare(
    {"RW": rb.RandomWalk(), "OLS-VAR(2)": rb.OLSVAR(lags=2), "BVAR(2)": bvar},
    data,
    origins=rb.Origins(start="2010Q1", end="2019Q3"),
    window=rb.Rolling(size=120),
    horizons=[1, 4],
    baseline="RW",
)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 200)
cols = ["origins", "dlpd", "dlpd_mcse", "crps_rel", "dm_pvalue", "tests_valid", "in_mcs_25", "converged"]
print(res.table()[cols].round(3))
v = res.verdict()
print(v[["model", "reference", "axis", "horizon", "verdict"]])
for r in v.itertuples():
    print(f"{r.model} vs {r.reference} h={r.horizon}: {r.reasons}")
