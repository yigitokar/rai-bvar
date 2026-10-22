"""Empirical Bayes for the MCMC engine (DESIGN §6.5; D-12, E5, E25; WP18).

The plug-in comes from the Phase 1 ``AnalyticFit`` of the conjugate twin at ``cross = contemp =
1`` (so it equals the Phase 1 EB mode), from Chan's (2022) structural NIG marginal likelihood for
``form="structural"`` with ``cross ≠ 1``, and from the twin with a recorded note for
``form="reduced"`` with ``cross ≠ 1``.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

import rai_bvar as rb
from rai_bvar.engines import structural_nig
from rai_bvar.engines.analytic import AnalyticFit
from rai_bvar.engines.mcmc import eb, resolve
from rai_bvar.priors.scales import window_scales
from rai_bvar.priors.structural import structural_arrays

LAGS = 2
RUNTIME = rb.Runtime(chunk=10)


def mcmc_model(vol=None, form="structural", **prior):
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form=form, **prior)),
        vol=rb.SV() if vol is None else vol,
        lags=LAGS,
        mcmc=rb.MCMCSettings(chains=2, draws=10, burn=0),
    )


def phase1(data, **prior):
    """The Phase 1 conjugate EB fit (``BVAR()`` with the given Minnesota settings)."""
    model = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(**prior)), lags=LAGS)
    assert model.engine == "analytic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return AnalyticFit(model, data, None)


@pytest.fixture(scope="module")
def data(sim_data_stationary):
    return sim_data_stationary.tail(120)


@pytest.mark.parametrize(
    ("form", "vol"),
    [("structural", rb.SV()), ("structural", rb.Constant("ig")), ("reduced", rb.SV())],
    ids=["structural-SV", "structural-ig", "reduced-SV"],
)
def test_eb_plug_in_equals_the_phase1_eb_mode_at_cross_1(data, form, vol):
    """At ``cross = 1`` the plug-in is the Phase 1 EB mode (1e-8), and ``resolve`` builds the
    prior arrays from it (DESIGN §6.5 steps 1 and 4)."""
    model = mcmc_model(vol, form=form)
    out = eb.eb_hyper(model, data)
    ref = phase1(data)
    assert out.hyper["eb_method"] == "conjugate_twin" and out.hyper["eb_free"] == ("lambda",)
    np.testing.assert_allclose(out.hyper["lambda"], ref.hyper["lambda"], rtol=1e-8)
    np.testing.assert_allclose(out.hyper["alpha"], ref.hyper["alpha"], rtol=1e-8)
    np.testing.assert_allclose(out.hyper["psi"], ref.hyper["psi"], rtol=1e-8)
    assert out.converged is ref.converged is True and out.note is None
    r = resolve.resolve(model, data, RUNTIME)
    assert r.hyper["lambda"] == out.hyper["lambda"] and r.eb_converged is True
    assert r.notes == () and "eb_note" not in r.hyper


def test_eb_over_lambda_alpha_and_psi_is_plugged_into_the_builders(data):
    """With ``hyper=("lambda", "alpha", "psi")`` the twin estimates all three; the window scales
    and δ arrays use the EB ψ, λ and α (DESIGN §6.5 step 4)."""
    free = ("lambda", "alpha", "psi")
    model = mcmc_model(hyper=free)
    out = eb.eb_hyper(model, data)
    ref = phase1(data, hyper=free)
    assert out.hyper["eb_free"] == free
    for key in free:
        np.testing.assert_allclose(out.hyper[key], ref.hyper[key], rtol=1e-8)
    r = resolve.resolve(model, data, RUNTIME)
    np.testing.assert_array_equal(r.hyper["psi"], out.hyper["psi"])
    scales = window_scales(data, model.mean.prior, LAGS, "c", psi=out.hyper["psi"])
    values = {"lambda": out.hyper["lambda"], "alpha": out.hyper["alpha"]}
    coef, _ = structural_arrays(scales, model.mean.prior, values, n=3, lags=LAGS, n_det=1, mode="I")
    np.testing.assert_array_equal(r.prior.eq.coef.prec0, coef.prec0)
    ar = window_scales(data, model.mean.prior, LAGS, "c")
    assert not np.array_equal(r.hyper["psi"], ar.psi)  # the EB ψ, not the AR default


def test_structural_cross_not_1_uses_eb_structural(data, monkeypatch):
    """``form="structural"``, ``cross ≠ 1``: Chan's (2022) NIG EB from the twin's mode, with
    ``cross`` fixed (DESIGN §6.5 step 2)."""
    calls = []
    real = structural_nig.eb_structural

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(structural_nig, "eb_structural", spy)
    model = mcmc_model(cross=0.5)
    out = eb.eb_hyper(model, data)
    assert len(calls) == 1
    (d, prior, lags, det), kwargs = calls[0]
    twin = phase1(data)
    assert d is data and prior is model.mean.prior and (lags, det) == (LAGS, "c")
    assert kwargs["start"]["lambda"] == twin.hyper["lambda"]
    expected = real(data, model.mean.prior, LAGS, "c", start=twin.hyper)
    assert out.hyper["eb_method"] == "structural_nig"
    assert out.hyper["lambda"] == expected.hyper["lambda"] and out.converged is expected.converged
    assert abs(out.hyper["lambda"] - twin.hyper["lambda"]) > 1e-4  # a different objective
    # the mode beats the twin's start under Chan's objective
    hp = model.mean.prior.hyperpriors.lambda_

    def objective(lam):
        log_ml = structural_nig.structural_log_ml(
            data, model.mean.prior, {"lambda": lam}, LAGS, "c"
        )
        return log_ml + hp.logpdf(lam)

    assert objective(out.hyper["lambda"]) >= objective(twin.hyper["lambda"])
    r = resolve.resolve(model, data, RUNTIME)
    assert r.hyper["eb_method"] == "structural_nig" and r.hyper["lambda"] == out.hyper["lambda"]


def test_reduced_cross_not_1_records_the_note_and_does_not_raise(data):
    """``form="reduced"``, ``cross ≠ 1``: the ``cross = 1`` twin's mode with a recorded note in
    ``hyper["eb_note"]`` and the notes, never a warning (DESIGN §6.5 step 3; fixes E5)."""
    model = mcmc_model(form="reduced", cross=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the note is not a warning
        out = eb.eb_hyper(model, data)
        r = resolve.resolve(model, data, RUNTIME)
    twin = phase1(data)
    assert out.note == eb.REDUCED_CROSS_NOTE == "EB at cross=1 plugged into the reduced-form prior"
    assert (
        out.hyper["lambda"] == twin.hyper["lambda"] and out.hyper["eb_method"] == "conjugate_twin"
    )
    assert r.hyper["eb_note"] == eb.REDUCED_CROSS_NOTE and r.notes == (eb.REDUCED_CROSS_NOTE,)
    assert eb.REDUCED_CROSS_NOTE not in r.warning_inputs["eb_warnings"]
    assert r.hyper["cross"] == 0.5  # the reduced-form prior keeps its own cross


def test_eb_converged_flag_and_messages_propagate(data, monkeypatch):
    """The twin's (or Chan EB's) ``converged`` flag and optimisation messages reach the outcome
    and, through ``resolve``, ``eb_converged`` and the warning inputs (DESIGN §6.5 step 4)."""

    class Stuck(AnalyticFit):
        @property
        def converged(self) -> bool:
            return False

        def __init__(self, *args):
            super().__init__(*args)
            self.warnings.append("hyperparameter optimisation did not converge: (test double)")

    monkeypatch.setattr(eb, "AnalyticFit", Stuck)
    out = eb.eb_hyper(mcmc_model(), data)
    assert out.converged is False
    assert "hyperparameter optimisation did not converge: (test double)" in out.warnings
    r = resolve.resolve(mcmc_model(), data, RUNTIME)
    assert r.eb_converged is False and r.hyper["eb_converged"] is False
    assert (
        "hyperparameter optimisation did not converge: (test double)"
        in (r.warning_inputs["eb_warnings"])
    )
    monkeypatch.undo()

    real = structural_nig.eb_structural

    def stuck_chan(*args, **kwargs):
        res = real(*args, **kwargs)
        return dataclasses.replace(res, converged=False, warnings=("structural EB stuck",))

    monkeypatch.setattr(eb.structural_nig, "eb_structural", stuck_chan)
    out = eb.eb_hyper(mcmc_model(cross=0.5), data)
    assert out.converged is False and "structural EB stuck" in out.warnings


def test_twin_setting_warnings_are_dropped_and_psi_notes_kept(simulate_var):
    """The twin's 2020 warning is recomputed by the MCMC fit (``SV(outliers=True)`` counts as a
    treatment), so it is not propagated; its ψ notes are (DESIGN §6.5 step 4)."""
    index = pd.period_range("2017Q1", periods=14, freq="Q")
    frame = pd.DataFrame(simulate_var(14, 3), index=index, columns=["gdp", "prices", "rate"])
    short = rb.VARData.from_frame(frame, 1)  # covers 2020; AR(2) residuals have 9 d.o.f.
    twin = phase1(short)
    assert any("covers 2020" in w for w in twin.warnings)
    assert any("degrees of" in w for w in twin.warnings)
    out = eb.eb_hyper(mcmc_model(rb.SV(outliers=True)), short)
    assert not any("covers 2020" in w for w in out.warnings)
    assert any("degrees of" in w for w in out.warnings)


def test_hierarchy_fixed_runs_no_eb(data, monkeypatch):
    """``hierarchy="fixed"``: the prior's values, no EB call, ``eb_converged`` is ``None``."""

    def boom(*args):
        raise AssertionError("EB must not run with hierarchy='fixed'")

    monkeypatch.setattr(eb, "eb_hyper", boom)
    r = resolve.resolve(mcmc_model(hierarchy="fixed", lambda_=0.3), data, RUNTIME)
    assert r.hyper["lambda"] == 0.3 and r.hyper["hierarchy"] == "fixed"
    assert r.eb_converged is None and "eb_method" not in r.hyper


def test_eb_hyper_rejects_other_models(data):
    with pytest.raises(ValueError, match="MCMC compositions"):
        eb.eb_hyper(rb.BVAR(), data)
    with pytest.raises(ValueError, match="hierarchy='eb'"):
        eb.eb_hyper(mcmc_model(hierarchy="fixed"), data)
    with pytest.raises(TypeError, match="VARData"):
        eb.eb_hyper(mcmc_model(), data.values)


def test_conjugate_twin_is_the_phase1_composition():
    """The twin keeps every Minnesota setting except form, cross, contemp and hierarchy."""
    model = mcmc_model(cross=0.4, contemp=2.0, lambda_=0.3, alpha=1.5, var_const=1e5, mean=0.9)
    twin = eb.conjugate_twin(model)
    p = twin.mean.prior
    assert twin.engine == "analytic" and twin.vol == rb.Constant("iw")
    assert (p.form, p.cross, p.contemp, p.hierarchy) == ("conjugate", 1.0, 1.0, "eb")
    assert (p.lambda_, p.alpha, p.var_const, p.mean) == (0.3, 1.5, 1e5, 0.9)
    assert (twin.lags, twin.deterministic, twin.outliers) == (LAGS, "c", None)
