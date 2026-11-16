"""Bitwise A/B of the Phase 2 user paths across a Phase 3 work package (DESIGN P3 §2, E118).

Committed draws cannot be bitwise across CPU models (P2 §9.10), so every Phase 3 work package
that touches the engine (WP26–WP34) records a same-machine comparison of its parent commit and
its head. On one simulated dataset the tool runs (each case fitted with a short fixed budget and
forecast; every array of ``fit.posterior``, of the forecast and of ``fit.hyper`` hashed):

* **every Phase 2 composition** of ``validation.joint.COMPOSITIONS`` that the public API can build
  (all but row 17, ``zero_sv``, which has no public mean), at ``hierarchy="fixed"``, built from the
  registry's own definitions so that no row can fall out of the gate (``ROWS`` maps each case to
  its row; ``tests/mcmc/unit/test_contracts_p3.py`` checks the coverage);
* **options** a composition row does not pin: ``const_path="cholesky"`` (the branch E65 changes),
  ``gram="batched"`` on the two paths that read the Gram layout (the weighted structural step and
  the reduced-form CCCM step), ``store="paths"``, ``deterministic="ct"``, ``SV(exact=False)``,
  ``SV(mixture="ksc7")``, ``SV(tails="t", t_scale="unit")`` and a structural ``cross != 1`` at
  ``hierarchy="fixed"``;
* **the EB plug-in**: the structural ``cross != 1`` compositions at the default ``hierarchy="eb"``
  (Constant, SV, and Constant with SOC/SUR rows, through ``eb_hyper -> eb_structural``) and the
  default MCMC path at ``cross = 1`` (the DGP study's arms), with ``fit.hyper`` in the digest;
* **the analytic engine**: ``AnalyticFit`` at ``hierarchy`` ``fixed``, ``eb`` and ``mh`` (``log_ml``,
  ``hyper``, the MH chains and the forecast's ``paths``, ``cond_mean``, ``cond_chol``), and one
  forecast-mode ``ordering_sensitivity`` of an analytic model (its dataset's arrays).

    python -I tools/phase3_bitwise_ab.py run OUT.json        # on the parent, then on the head
    python -I tools/phase3_bitwise_ab.py compare A.json B.json

``compare`` exits non-zero and lists the cases whose digests differ; ``run`` exits non-zero when
two MCMC cases share a posterior digest (a case that exercises nothing of its own). Run it in a
checkout whose ``src`` is the code under test (``uv run`` in that worktree); a PR records the head
commit, the machine and the sha256 of both JSON files with the result.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd

ROWS: dict[str, int | None] = {}  # case -> Phase 2 composition row (None: an option or path case)


def _data():
    import rai_bvar as rb

    rng = np.random.default_rng(20261025)
    n, T = 3, 66
    B = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.1 + B @ y[t - 1] + rng.standard_normal(n) * np.exp(0.3 * np.sin(t / 7.0))
    index = pd.period_range("1980Q1", periods=T, freq="Q")
    frame = pd.DataFrame(y, index=index, columns=["a", "b", "c"])
    return rb.VARData.from_frame(frame, tcodes=1)


def _vol_of(d):
    """The public volatility of a ``validation.joint._Def`` row."""
    import rai_bvar as rb

    if d.persistent == "constant":
        return rb.Constant({"I": "ig", "II": "iw", "iw_equivalent": "iw_equivalent"}[d.mode])
    kw = {
        "rw": d.rw,
        "outliers": d.transitory in ("outlier", "outlier_t"),
        "tails": "t" if d.transitory in ("t", "outlier_t") else "normal",
    }
    if d.sigma2_prior == "inv_gamma":
        kw["sigma2_prior"] = rb.sv_priors.InvGamma(3.0, 0.5)
    if d.h0 == "fixed_b0":
        kw["h0"] = 2.0
    return rb.SV(**kw)


def _models():
    import rai_bvar as rb
    from rai_bvar.validation import joint

    M = rb.Minnesota
    settings = rb.MCMCSettings(chains=2, draws=40, burn=30, seed=5)

    def bv(vol, form, lags=1, hierarchy="fixed", mcmc=settings, det="c", **prior):
        return rb.BVAR(
            mean=rb.Linear(prior=M(form=form, hierarchy=hierarchy, **prior)),
            vol=vol,
            lags=lags,
            deterministic=det,
            mcmc=mcmc,
        )

    cases = {}
    ROWS.clear()
    for name, d in joint._COMPOSITIONS.items():
        if d.mean == "zero":
            continue  # row 17: no public mean
        form = "structural" if d.mean == "linear_structural" else "reduced"
        dummies = {"soc": 1.0, "sur": 1.0} if d.dummies else {}
        cases[name] = bv(_vol_of(d), form, lags=1 + d.number % 2, **dummies)
        ROWS[name] = d.number

    def opt(**changes):
        return rb.MCMCSettings(chains=2, draws=40, burn=30, seed=5, **changes)

    options = {
        "opt-const-path-cholesky": bv(
            rb.Constant("ig"), "structural", mcmc=opt(const_path="cholesky")
        ),
        # gram is read only by the weighted structural step and the CCCM step (not by eigen etc.)
        "opt-gram-batched-structural-sv": bv(rb.SV(), "structural", mcmc=opt(gram="batched")),
        "opt-gram-batched-reduced": bv(rb.Constant("ig"), "reduced", 2, mcmc=opt(gram="batched")),
        "opt-store-paths": bv(rb.SV(), "structural", mcmc=opt(store="paths")),
        "opt-det-ct": bv(rb.Constant("ig"), "structural", det="ct"),
        "opt-sv-inexact": bv(rb.SV(exact=False), "structural"),
        "opt-sv-ksc7": bv(rb.SV(mixture="ksc7"), "reduced"),
        "opt-sv-t-unit": bv(rb.SV(tails="t", t_scale="unit"), "structural"),
        "opt-sv-cross-fixed": bv(rb.SV(), "structural", cross=0.5),
        # the EB plug-in through structural_nig.eb_structural (cross != 1; E118)
        "eb-structural-constant-cross": bv(rb.Constant("ig"), "structural", 2, "eb", cross=0.5),
        "eb-structural-sv-cross": bv(rb.SV(), "structural", 1, "eb", cross=0.5),
        "eb-structural-constant-cross-dummies": bv(
            rb.Constant("ig"), "structural", 1, "eb", cross=0.5, soc=1.0, sur=1.0
        ),
        # the default MCMC user path (form auto -> structural, hierarchy eb, cross = 1)
        "eb-default-sv": rb.BVAR(vol=rb.SV(), lags=2, mcmc=settings),
    }
    mh = rb.MHSettings(draws=200, burn=100, chains=2, seed=3)
    for h in ("fixed", "eb", "mh"):
        options[f"analytic-{h}"] = rb.BVAR(
            mean=rb.Linear(prior=M(form="conjugate", hierarchy=h, mh=mh, soc=1.0)), lags=2
        )
    options["ordering-analytic"] = rb.BVAR(mean=rb.Linear(prior=M(form="conjugate")), lags=1)
    for name in options:
        ROWS[name] = None
    cases.update(options)
    return cases


def _digest_arrays(items) -> str:
    h = hashlib.sha256()
    for name, arr in items:
        arr = np.ascontiguousarray(np.asarray(arr))
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        # object arrays (e.g. a None hyperparameter) would hash memory addresses: hash their repr
        h.update(repr(arr.tolist()).encode() if arr.dtype == object else arr.tobytes())
    return h.hexdigest()


def _hyper_items(hyper) -> list:
    return [(f"hyper/{k}", np.asarray(hyper[k])) for k in sorted(hyper)]


def _ordering_digest(model, data) -> dict[str, str]:
    import rai_bvar as rb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rb.ordering_sensitivity(
            model, data, n_perm=3, mode="forecast", horizons=(1, 2), noise_reps=1, draws=200
        )
    ds = res.dataset
    return {"ordering": _digest_arrays(sorted((k, ds[k].values) for k in ds.data_vars))}


def _fit_digest(name, model, data) -> dict[str, str]:
    if name.startswith("ordering-"):
        return _ordering_digest(model, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(data, seed=11)
        fc = fit.forecast(3, draws=40, seed=2)
    ds = fc.dataset
    fcast = sorted((k, ds[k].values) for k in ds.data_vars)
    out = {"forecast": _digest_arrays(fcast), "hyper": _digest_arrays(_hyper_items(fit.hyper))}
    if model.engine == "analytic":
        fixed = [("log_ml", np.asarray(fit.log_ml()))]
        if fit.mh is not None:
            fixed.append(("mh/chains", np.asarray(fit.mh.chains)))
        out["fit"] = _digest_arrays(fixed)
        return out
    post = []
    for node in fit.posterior.subtree:
        for k, var in node.ds.data_vars.items():
            post.append((f"{node.path}/{k}", var.values))
    post.sort(key=lambda kv: kv[0])
    out["posterior"] = _digest_arrays(post)
    return out


def run(out: str) -> None:
    data = _data()
    result = {name: _fit_digest(name, m, data) for name, m in _models().items()}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    rows = sum(r is not None for r in ROWS.values())
    print(f"{len(result)} cases ({rows} composition rows) written to {out}")
    seen: dict[str, str] = {}
    duplicates = []
    for name, digest in sorted(result.items()):
        post = digest.get("posterior")
        if post is not None and post in seen:
            duplicates.append((seen[post], name))
        seen.setdefault(post, name)
    for a, b in duplicates:
        print(f"DUPLICATE posterior digest: {a} and {b}")
    if duplicates:
        sys.exit(2)


def compare(a: str, b: str) -> int:
    with open(a) as fh:
        A = json.load(fh)
    with open(b) as fh:
        B = json.load(fh)
    bad = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    for k in bad:
        print(f"DIFFERENT: {k}")
    print(f"{len(set(A) & set(B)) - len(bad)} of {len(set(A) | set(B))} cases bitwise equal")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
