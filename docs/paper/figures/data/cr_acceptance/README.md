# Per-origin scores of the Clark–Ravazzolo acceptance run

`cr_acceptance_per_origin.csv` holds the per-origin, per-horizon scores of the five main-stage
arms of the pre-registered Clark–Ravazzolo acceptance run (OUTLINE TODO 5). FigF1
(`make_fig_cr_per_origin.py`) and FigF3 (`make_fig_cr_cumulative.py`) are drawn from it.
`cr_acceptance_per_origin.meta.json` is its side file: the source, the git state, the config and
the result of the check against the report.

## Source

| | |
| :- | :- |
| Run | `benchmarks/clark_ravazzolo.py benchmarks/reports/clark_ravazzolo_acceptance --config acceptance --n-jobs 10 --cache ~/.cache/rai-bvar-cr-c823d20`, run at **c823d20** (`c823d207e0525aed98f427f437d687f7f131c92c`) with a clean tree (`versions.git_dirty = false` in the report) |
| Config | `acceptance`: 4 chains × 20,000 kept draws, burn-in 1000; 8000 forecast draws; `lpd_method = "integrated"`; seed 0 |
| Report | `benchmarks/reports/clark_ravazzolo_acceptance.json`, committed in 65dd212 (main repository; on `paper-draft` since 75db8a7), sha256 `d5c3ec8d6911a4d26940b7351e62f45b8d89cf01d3bcc9958b24772118e935e3` |
| Score cache | `~/.cache/rai-bvar-cr-c823d20` (the run's own cache; read only, see below) |
| Code used for the export | a clean clone of the main repository checked out at c823d20 (detached, `git status` empty), so `benchmarks/clark_ravazzolo.py` and `src/rai_bvar` are the run's own; the export checks after import that both come from that checkout. `rai_bvar` 0.0.1, as in the report's `versions` |
| Export script | `docs/paper/figures/make_data_cr_per_origin.py` (paper worktree, branch `paper-draft`, base 75db8a7; uncommitted at export time) |
| Exported | 2026-10-23 |

Command (from `docs/paper/figures/`):

```
PYTHONDONTWRITEBYTECODE=1 /Users/yigitokar/rai-bvar-paper/.venv/bin/python make_data_cr_per_origin.py \
    --cache ~/.cache/rai-bvar-cr-c823d20 --repo <clean checkout of c823d20> \
    --check /Users/yigitokar/rai-bvar-paper/benchmarks/reports/clark_ravazzolo_acceptance.json
```

## How the scores were read

- **Same cache keys as the run.** The export rebuilds the main-stage backtests exactly as the
  run's script builds them, with the same models (`cr.models(cfg.mcmc)`), the same `MomentsProbe`
  wrapper, the 106 origins 1985Q1–2011Q2, seed 0, 8000 draws and `lpd_method="integrated"`. It
  calls `backtest(..., cache=...)` with `Cache.put`, `Cache.put_state` and every model fit
  disabled, so an origin missing from the cache would have stopped the export with an error.
  All 5 × 106 = 530 origins were read from the cache (`res.cached == 106` for every arm).
- **The cache was not written.** A marker file was touched before the export.
  `find ~/.cache/rai-bvar-cr-c823d20 -newer <marker>` listed nothing afterwards.
- **Gaussian scores.** The Gaussian-approximation log scores (`lps_gauss`) are recomputed from
  the cached forecast moments (`<cache>/moments`) with the run script's own `gaussian_lps`, as the
  report does.

## Check against the report

`--check` re-reads the CSV and recomputes 2236 report values with the formulas of the run
script's `relative_table` (l.479 at c823d20). It covers `n`, mean, median and the top three
origins by |d| (576 origin and value cells) for every variable and horizon. It covers the primary,
path-conditioned and Gaussian ΔLPS and ΔLPD of `relative.{full,pre_crisis}[arm]` for the four SV
arms against Constant, plus `variants[variant][ref]`, `converged` and `n_origins`. Tolerance:
1e-9. **Result: every value reproduced, maximum absolute difference 0** (bitwise). The floats are
written with 17 significant digits and read back with `float_precision="round_trip"`.

The three origins that drive criterion 4's unemployment cell were also checked by hand. They are
the per-origin ΔLPS (SV(rw) − Constant, primary path-integrated estimator) at UR, h = 8, full
sample, against `relative.full['SV(rw)'].variables.UR['8'].dlps_top`:

| origin | export | report |
| :- | -: | -: |
| 2007Q4 | +7.594654597593976 | +7.594654597593976 |
| 2008Q1 | +6.626345271154726 | +6.626345271154726 |
| 2008Q2 | +6.51530354616955 | +6.51530354616955 |

The cell means also match: UR h = 1 +0.160 → h = 8 +0.273 (median +0.140); TBILL h = 1 +0.377 →
h = 8 −0.195; pre-crisis GDP h = 8 +0.105 and INFL h = 8 +0.307. The path-conditioned UR h = 8 is
+1.309, and the joint h = 1 ΔLPD is +0.819. Every arm converged at 106 of 106 origins.

## Columns

There is one row per (model, origin, horizon, variable): 5 models × 106 origins × horizons
{1, 2, 4, 8} × {GDP, INFL, UR, TBILL, `joint`} = 10,600 rows.

| column | meaning (cache variable) |
| :- | :- |
| `model` | Constant, SV(rw), SV, SV(rw,t10), AR-SV(0.9) (reduced form, identity ordering) |
| `origin`, `horizon` | forecast origin (quarter) and horizon |
| `converged` | 1 if the origin's fit passed the convergence rule (`converged`) |
| `variable` | GDP, INFL, UR, TBILL, or `joint` |
| `lpd` | primary RB log score, path-integrated (`lpd` for joint rows, `lpd_marginal` for marginal rows) |
| `lpd_conditioned` | path-conditioned RB log score from the same draws, secondary (`lpd_conditioned` / `lpd_marginal_conditioned`) |
| `lps_gauss` | Gaussian-approximation log score from the forecast moments |
| `lpd_mcse` | delta-method MCSE of the primary score (`lpd_mcse` / `lpd_marginal_mcse`) |
| `lpd_mcse_conditioned` | MCSE of the path-conditioned joint score (`lpd_mcse_conditioned`; joint rows only) |
| `crps`, `sq_err` | CRPS and squared error (marginal rows only) |

At h = 1 the two RB estimators coincide by construction (D-34).

The structural pair (`Constant [structural]`, `SV(rw) [structural]`) and the 24-ordering stage
are not exported. The figures use the report for those (`structural`, `orderings`).
