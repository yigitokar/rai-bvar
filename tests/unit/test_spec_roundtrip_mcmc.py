"""``spec()`` / ``from_spec`` for MCMC compositions (DESIGN §11.5; fixes E13).

* Property test (hypothesis): ``from_spec(m.spec()) == m`` through plain JSON, with a stable
  spec hash, name and axes, over random valid compositions (every form, every ``Constant``
  prior, ``SV`` with random nested priors, ``MCMCSettings``).
* Spec layout: the SV dict carries ``"class": "SV"`` and nested ``{"class": …}`` priors; a
  ``Constant`` dict has no tag; ``mcmc`` appears exactly on MCMC models; ``contemp`` only when
  it differs from 1.
* ``from_spec`` defaults missing Phase 2 fields and rejects unknown volatility classes.
* The hash is a cache key: stable across processes and distinct across settings.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import rai_bvar as rb
from rai_bvar.models import spec_hash

P = rb.sv_priors

positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
real = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
numeric = st.sampled_from([float, np.float64, np.float32])


@st.composite
def minnesota(draw, form: str) -> rb.Minnesota:
    kwargs = {
        "lambda_": draw(numeric)(draw(positive)),
        "alpha": draw(st.sampled_from([2.0, 1.0, 1.5])),
        "var_const": draw(st.sampled_from([1e7, 1e5])),
        "hierarchy": draw(st.sampled_from(["fixed", "eb"])),
        "psi": draw(st.sampled_from(["ar", (1.0, 2.0, 0.5)])),
        "mean": draw(st.sampled_from([None, 1.0, {"gdp": 0.9, "rate": 0.5}])),
    }
    if form in ("structural", "reduced"):
        kwargs["cross"] = draw(st.sampled_from([1.0, 0.5, 0.2]))
        kwargs["contemp"] = draw(st.sampled_from([1.0, 0.25, 4.0]))
        if kwargs["hierarchy"] == "eb" and kwargs["cross"] != 1.0 and form == "structural":
            kwargs["hyper"] = draw(st.sampled_from(["auto", ("lambda", "alpha")]))
        else:
            kwargs["hyper"] = draw(
                st.sampled_from(["auto", ("lambda", "psi"), ("lambda", "alpha")])
            )
    return rb.Minnesota(form=form, **kwargs)


@st.composite
def sv_kwargs(draw) -> dict:
    """Random valid ``SV`` keywords. Settings the variant ignores (DESIGN §6.3) are drawn too,
    half of the time, with any valid value."""
    rw = draw(st.booleans())
    outliers = draw(st.booleans())
    tails = draw(st.sampled_from(["normal", "t"]))
    kwargs = {
        "rw": rw,
        "outliers": outliers,
        "tails": tails,
        "mixture": draw(st.sampled_from(["ocsn10", "ksc7"])),
        "offset": draw(st.sampled_from(["relative", 0.0, 1e-4])),
        "exact": draw(st.booleans()),
        "asis": draw(st.booleans()),
        "sigma2_prior": draw(
            st.one_of(
                st.none(),
                st.builds(P.ScaledChi2, positive),
                st.builds(P.InvGamma, positive, positive),
            )
        ),
    }
    unused = _unused(kwargs)  # before mu_prior is drawn: mu_var is always drawn without rw
    t_scale = draw(st.sampled_from(["standardized", "unit"]))
    low = draw(st.integers(3 if tails == "t" and t_scale == "standardized" else 1, 20))
    candidates = {
        "rw_h0_var": positive,
        "mu_prior": st.one_of(st.just("data"), st.builds(P.Normal, real, positive)),
        "mu_var": positive,
        "phi_prior": st.builds(P.Beta, positive, positive),
        "h0": st.one_of(st.just("stationary"), positive),
        "outlier_max": st.integers(2, 30),
        "outlier_prior": st.one_of(
            st.none(),
            st.builds(P.OutlierPrior, positive, positive),
            st.builds(P.Beta, positive, positive),
        ),
        "t_scale": st.just(t_scale),
        "nu_prior": st.integers(low, 60).map(lambda high: P.DiscreteUniform(low, high)),
    }
    for name, strategy in candidates.items():
        if name not in unused or draw(st.booleans()):
            kwargs[name] = draw(strategy)
    return kwargs


def _unused(kwargs: dict) -> set[str]:
    """Settings the variant ignores (DESIGN §6.3, §3.8), written out independently of specs.py."""
    unused = set()
    if kwargs["rw"]:
        unused |= {"mu_prior", "mu_var", "phi_prior", "h0"}
    else:
        unused.add("rw_h0_var")
        if kwargs.get("mu_prior", "data") != "data":
            unused.add("mu_var")
    if not kwargs["outliers"]:
        unused |= {"outlier_max", "outlier_prior"}
    if kwargs["tails"] != "t":
        unused |= {"nu_prior", "t_scale"}
    return unused


def sv() -> st.SearchStrategy[rb.SV]:
    return sv_kwargs().map(lambda kwargs: rb.SV(**kwargs))


mcmc_settings = st.one_of(
    st.none(),
    st.builds(
        rb.MCMCSettings,
        chains=st.integers(1, 8),
        draws=st.integers(1, 20_000),
        burn=st.integers(0, 5_000),
        thin=st.integers(1, 5),
        warm_burn=st.one_of(st.none(), st.integers(0, 1_000)),
        seed=st.integers(0, 2**40),
        store=st.sampled_from(["params", "paths"]),
        gram=st.sampled_from(["auto", "packed", "packed_tiled", "batched"]),
        const_path=st.sampled_from(["auto", "cholesky"]),
        rhat_max=st.sampled_from([1.01, 1.05]),
    ),
)


@st.composite
def mcmc_bvar(draw) -> rb.BVAR:
    form = draw(st.sampled_from(["auto", "structural", "reduced"]))
    if form == "reduced":
        vol = draw(
            st.one_of(
                st.sampled_from([rb.Constant(), rb.Constant("ig"), rb.Constant("iw_equivalent")]),
                sv(),
            )
        )
    elif form == "structural":
        vol = draw(
            st.one_of(
                st.sampled_from([rb.Constant(p) for p in ("auto", "ig", "iw", "iw_equivalent")]),
                sv(),
            )
        )
    else:  # "auto" resolves to structural with a non-conjugate volatility
        vol = draw(
            st.one_of(st.sampled_from([rb.Constant("ig"), rb.Constant("iw_equivalent")]), sv())
        )
    prior = draw(minnesota("structural" if form == "auto" else form))
    if form == "auto":
        prior = dataclasses.replace(prior, form="auto", cross=1.0, contemp=1.0)
    return rb.BVAR(
        mean=rb.Linear(prior=prior),
        vol=vol,
        lags=draw(st.integers(1, 4)),
        deterministic=draw(st.sampled_from(["c", "ct", "n"])),
        mcmc=draw(mcmc_settings),
    )


def _assert_round_trip(model: rb.BVAR) -> None:
    payload = json.dumps(model.spec())  # plain JSON, no ``default=``
    rebuilt = rb.BVAR.from_spec(json.loads(payload))
    assert rebuilt == model
    assert spec_hash(rebuilt) == spec_hash(model)
    assert rebuilt.name == model.name and rebuilt.axes == model.axes
    assert json.dumps(rebuilt.spec()) == payload
    assert rb.BVAR.from_spec(json.loads(json.dumps(rebuilt.spec()))) == model


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(mcmc_bvar())
def test_from_spec_inverts_spec_for_random_mcmc_compositions(model):
    assert model.engine == "mcmc"
    _assert_round_trip(model)


@settings(max_examples=50, deadline=None)
@given(sv())
def test_random_sv_specs_round_trip(vol):
    assert rb.SV.from_spec(json.loads(json.dumps(vol.to_spec()))) == vol


@settings(max_examples=200, deadline=None)
@given(sv_kwargs())
def test_ignored_sv_settings_leave_one_spec(kwargs):
    """``SV`` with ignored settings is the same model (spec, hash, label) as without them."""
    vol = rb.SV(**kwargs)
    used = rb.SV(**{k: v for k, v in kwargs.items() if k not in _unused(kwargs)})
    assert vol == used and vol.to_spec() == used.to_spec() and vol.label == used.label
    assert spec_hash(rb.BVAR(vol=vol)) == spec_hash(rb.BVAR(vol=used))


EXAMPLES = {
    "mode-i": rb.BVAR(vol=rb.Constant("ig"), lags=2),
    "mode-ii": rb.BVAR(mean=rb.Linear(rb.Minnesota(form="structural")), vol=rb.Constant("iw")),
    "iw-equivalent": rb.BVAR(vol=rb.Constant("iw_equivalent")),
    "sv": rb.BVAR(vol=rb.SV(), lags=4),
    "svo-t": rb.BVAR(vol=rb.SV(outliers=True, tails="t", outlier_prior=P.Beta(1.0, 50.0))),
    "sv-rw": rb.BVAR(vol=rb.SV(rw=True, sigma2_prior=P.InvGamma(3.0, 0.1))),
    "sv-parity": rb.BVAR(
        vol=rb.SV(
            mu_prior=P.Normal(0.0, 100.0),
            phi_prior=P.Beta(5.0, 1.5),
            sigma2_prior=P.ScaledChi2(0.1),
            h0=1.0,
            offset=0.0,
            exact=False,
        )
    ),
    "reduced": rb.BVAR(
        mean=rb.Linear(rb.Minnesota(form="reduced", cross=0.2, contemp=0.5, hierarchy="fixed")),
        vol=rb.SV(outliers=True),
    ),
    "reduced-iweq": rb.BVAR(
        mean=rb.Linear(rb.Minnesota(form="reduced")), vol=rb.Constant("iw_equivalent")
    ),
    "settings": rb.BVAR(
        vol=rb.SV(), mcmc=rb.MCMCSettings(chains=2, draws=500, warm_burn=10, gram="batched")
    ),
}


@pytest.mark.parametrize("model", EXAMPLES.values(), ids=EXAMPLES.keys())
def test_from_spec_inverts_spec_for_every_mcmc_composition(model):
    _assert_round_trip(model)


def test_spec_layout_of_an_sv_model():
    spec = EXAMPLES["svo-t"].spec()
    assert list(spec) == ["class", "mean", "vol", "lags", "deterministic", "outliers", "mcmc"]
    assert spec["vol"]["class"] == "SV"
    assert spec["vol"]["outlier_prior"] == {"class": "Beta", "a": 1.0, "b": 50.0}
    assert spec["vol"]["nu_prior"] == {"class": "DiscreteUniform", "low": 3, "high": 40}
    assert spec["mcmc"] == dataclasses.asdict(rb.MCMCSettings())
    assert "contemp" not in spec["mean"]["prior"]


def test_spec_layout_of_constant_models():
    assert EXAMPLES["mode-ii"].spec()["vol"] == {"prior": "iw"}
    assert EXAMPLES["iw-equivalent"].spec()["vol"] == {"prior": "iw_equivalent"}
    assert EXAMPLES["mode-i"].spec()["vol"] == {"prior": "ig"}
    assert "mcmc" in EXAMPLES["mode-i"].spec() and "mcmc" not in rb.BVAR().spec()
    assert EXAMPLES["reduced"].spec()["mean"]["prior"]["contemp"] == 0.5


def test_from_spec_defaults_missing_phase2_fields():
    spec = json.loads(json.dumps(EXAMPLES["sv"].spec()))
    del spec["mcmc"]
    spec["vol"] = {"class": "SV"}
    assert rb.BVAR.from_spec(spec) == EXAMPLES["sv"]  # mcmc → MCMCSettings(), SV defaults
    spec["vol"] = {"prior": "ig"}  # no class tag: Constant (the Phase 1 layout)
    assert rb.BVAR.from_spec(spec).vol == rb.Constant("ig")


def test_from_spec_rejects_unknown_volatility_classes():
    spec = json.loads(json.dumps(EXAMPLES["sv"].spec()))
    spec["vol"] = {"class": "FactorSV", "factors": 2}
    with pytest.raises(ValueError, match="unknown volatility class 'FactorSV'"):
        rb.BVAR.from_spec(spec)


def test_spec_hashes_distinguish_mcmc_settings_and_sv_settings():
    variants = [
        rb.BVAR(vol=rb.SV()),
        rb.BVAR(vol=rb.SV(), mcmc=rb.MCMCSettings(seed=1)),
        rb.BVAR(vol=rb.SV(), mcmc=rb.MCMCSettings(t_bucket=16)),
        rb.BVAR(vol=rb.SV(), mcmc=rb.MCMCSettings(gram="batched")),
        rb.BVAR(vol=rb.SV(), mcmc=rb.MCMCSettings(const_path="cholesky")),
        rb.BVAR(vol=rb.SV(exact=False)),
        rb.BVAR(vol=rb.SV(offset=0.0)),
        rb.BVAR(vol=rb.SV(mu_var=10.000001)),
        rb.BVAR(vol=rb.SV(mixture="ksc7")),
        rb.BVAR(vol=rb.SV(rw=True)),
        rb.BVAR(vol=rb.Constant("ig")),
        rb.BVAR(mean=rb.Linear(rb.Minnesota(form="structural", contemp=0.5)), vol=rb.SV()),
    ]
    hashes = [spec_hash(m) for m in variants]
    assert len(set(hashes)) == len(hashes)


def test_mcmc_spec_hash_is_stable_across_processes():
    code = (
        "import rai_bvar as rb\n"
        "from rai_bvar.models import spec_hash\n"
        "m = rb.BVAR(vol=rb.SV(outliers=True, tails='t', mu_prior=rb.sv_priors.Normal(0, 100)),\n"
        "            mean=rb.Linear(rb.Minnesota(mean={'b': 1.0, 'a': 0.5})), lags=2,\n"
        "            mcmc=rb.MCMCSettings(chains=2))\n"
        "print(spec_hash(m))\n"
    )
    expected = spec_hash(
        rb.BVAR(
            vol=rb.SV(outliers=True, tails="t", mu_prior=P.Normal(0.0, 100.0)),
            mean=rb.Linear(rb.Minnesota(mean={"a": 0.5, "b": 1.0})),
            lags=2,
            mcmc={"chains": 2},
        )
    )
    env = {**os.environ, "PYTHONHASHSEED": "4321"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == expected
