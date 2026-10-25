"""Bitwise A/B of the Phase 2 MCMC compositions across a Phase 3 work package (DESIGN P3 §2).

Committed draws cannot be bitwise across CPU models (P2 §9.10), so every Phase 3 work package
that touches the engine (WP26–WP34) records a same-machine comparison of its parent commit and
its head: each Phase 2 MCMC composition is fitted with a short fixed budget on one simulated
dataset, forecast, and every array of ``fit.posterior`` and of the forecast is hashed.

    python -I tools/phase3_bitwise_ab.py run OUT.json        # on the parent, then on the head
    python -I tools/phase3_bitwise_ab.py compare A.json B.json

``compare`` exits non-zero and lists the compositions whose digests differ. Run it in a checkout
whose ``src`` is the code under test (``uv run`` in that worktree).
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd


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


def _models():
    import rai_bvar as rb

    M = rb.Minnesota
    settings = rb.MCMCSettings(chains=2, draws=40, burn=30, seed=5)

    def bv(vol, form, lags=1, **prior):
        return rb.BVAR(
            mean=rb.Linear(prior=M(form=form, hierarchy="fixed", **prior)),
            vol=vol,
            lags=lags,
            mcmc=settings,
        )

    return {
        "structural-constant-ig": bv(rb.Constant("ig"), "structural"),
        "structural-mode2": bv(rb.Constant("iw"), "structural", lags=2),
        "structural-iw-equivalent": bv(rb.Constant("iw_equivalent"), "structural"),
        "structural-sv": bv(rb.SV(), "structural", lags=2),
        "structural-sv-outliers": bv(rb.SV(outliers=True), "structural"),
        "structural-sv-t": bv(rb.SV(tails="t"), "structural"),
        "structural-sv-outliers-t": bv(rb.SV(outliers=True, tails="t"), "structural"),
        "structural-sv-rw": bv(rb.SV(rw=True), "structural"),
        "structural-sv-cross": bv(rb.SV(), "structural", cross=0.5),
        "structural-constant-dummies": bv(rb.Constant("ig"), "structural", soc=1.0, sur=1.0),
        "reduced-constant-ig": bv(rb.Constant("ig"), "reduced"),
        "reduced-iw-equivalent": bv(rb.Constant("iw_equivalent"), "reduced", lags=2),
        "reduced-sv": bv(rb.SV(), "reduced", lags=2),
        "reduced-sv-outliers-t": bv(rb.SV(outliers=True, tails="t"), "reduced"),
        "reduced-sv-rw": bv(rb.SV(rw=True), "reduced"),
        "reduced-constant-dummies": bv(rb.Constant("ig"), "reduced", soc=1.0),
    }


def _digest_arrays(items) -> str:
    h = hashlib.sha256()
    for name, arr in items:
        arr = np.ascontiguousarray(np.asarray(arr))
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _fit_digest(model, data) -> dict[str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(data, seed=11)
        fc = fit.forecast(3, draws=40, seed=2)
    post = []
    for node in fit.posterior.subtree:
        for name, var in node.ds.data_vars.items():
            post.append((f"{node.path}/{name}", var.values))
    post.sort(key=lambda kv: kv[0])
    ds = fc.dataset
    fcast = sorted((name, ds[name].values) for name in ds.data_vars)
    return {"posterior": _digest_arrays(post), "forecast": _digest_arrays(fcast)}


def run(out: str) -> None:
    data = _data()
    result = {name: _fit_digest(m, data) for name, m in _models().items()}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(f"{len(result)} compositions written to {out}")


def compare(a: str, b: str) -> int:
    with open(a) as fh:
        A = json.load(fh)
    with open(b) as fh:
        B = json.load(fh)
    bad = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    for k in bad:
        print(f"DIFFERENT: {k}")
    print(f"{len(set(A) & set(B)) - len(bad)} of {len(set(A) | set(B))} compositions bitwise equal")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
