# Listing 2 of the paper (docs/paper/sections/07_walkthrough.tex) shows lines 40-72; lines 8-36
# repeat the imports and the data block of ex_a.py.
# Run from the main repository: cd /Users/yigitokar/rai-bvar && uv run python <this file>,
# or through run_examples.py. Output captured at main-repo commit 4ac7969 (release 0.2.0,
# clean tree; out/env.json); stdout equals the d58c9ef capture except the version line.
# Differs from dossier 02 section 13.2 only by the two wall-clock prints, removed (not reproducible).

import platform

import numpy as np
import pandas as pd

import rai_bvar as rb

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 200)

# Same simulated data as ex_a.py
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
print(platform.platform(), "| rai-bvar", rb.__version__)

# MCMC engine: same Linear mean, stochastic volatility; form="auto" resolves to "structural".
mc = rb.MCMCSettings(chains=4, draws=1000, burn=1000)
sv = rb.BVAR(vol=rb.SV(), lags=2, mcmc=mc)
print(sv.engine, "|", sv.name, "|", sv.mean.prior.form, "|", sv.axes)
fit = sv.fit(data.upto("2019Q3"))
s = fit.summary()
print("prior mode:", s.attrs["prior_mode"], "| hyper:", s.attrs["hyper"], "| converged:", fit.converged)
print(s[["mean", "sd", "rhat", "ess_bulk"]].round(3))
fc = fit.forecast(horizon=4, draws=2000, seed=0)
print(fc, "| engine attr:", fc.dataset.attrs["engine"])
y_obs = data.values[-1]  # 2019Q4, transformed scale
print("joint LPD h=1 and MCSE:", fc.log_pred_density(y_obs, horizon=1, return_mcse=True))
print("marginal LPD (rate):", fc.log_pred_density(y_obs[2:], horizon=1, variables=["rate"]))
irf = fit.irf(horizon=8, draws=500, seed=0)
print(irf)
print(irf.median().sel(horizon=[0, 4, 8], shock="rate").round(4).to_pandas())

# Volatility-axis comparison: both arms structural Mode I, differing only in vol (R2).
models = {
    "Linear x Constant": rb.BVAR(vol=rb.Constant("ig"), lags=2, mcmc=mc),
    "Linear x SV": sv,
}
for k, m in models.items():
    print(k, m.axes)
res = rb.compare(
    models, data,
    origins=rb.Origins(start="2017Q1", end="2019Q3"),
    window=rb.Rolling(size=120),
    horizons=[1],
    draws=1000,
    baseline="Linear x Constant",
)
print(res.table()[["origins", "dlpd", "dlpd_mcse", "dm_pvalue", "tests_valid", "in_mcs_25", "converged"]].round(3))
v = res.verdict(axis="vol")
print(v[["model", "reference", "axis", "horizon", "verdict"]])
for r in v.itertuples():
    print(r.reasons)
