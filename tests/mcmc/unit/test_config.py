"""Static configuration (DESIGN §2.4), the §3.8 composition rules and the kernel registry."""

from __future__ import annotations

import functools
from typing import get_args

import numpy as np
import pytest

from rai_bvar.engines.mcmc import config as cfg
from rai_bvar.engines.mcmc import registry
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StoreStatic,
    StructStatic,
    VolStatic,
    bucket,
    scaling_mode,
    validate_composition,
)

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def make(**kw) -> StaticConfig:
    parts = {
        "shapes": Shapes.of(3, 2, 1, 64),
        "struct": StructStatic(),
        "mean": MeanStatic("linear_structural"),
        "coef": CoefPriorStatic(),
        "vol": VolStatic("sv"),
    }
    parts.update(kw)
    return StaticConfig(**parts)


# ------------------------------------------------------------------------------------- shapes


def test_shapes():
    s = Shapes.of(n=20, lags=4, n_det=1, T_pad=320)
    assert (s.k, s.q, s.P, s.P_x) == (81, 100, 5050, 3321)
    assert Shapes.of(1, 1, 0, 32).q == 1
    with pytest.raises(ValueError, match="k must equal"):
        Shapes(n=2, lags=1, n_det=1, k=4, q=4, T_pad=32)
    with pytest.raises(ValueError, match="q must equal"):
        Shapes(n=2, lags=1, n_det=1, k=3, q=3, T_pad=32)
    with pytest.raises(ValueError, match="999"):
        Shapes.of(1000, 1, 1, 32)
    with pytest.raises(TypeError):
        Shapes.of(2.0, 1, 1, 32)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Shapes.of(True, 1, 1, 32)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Shapes.of(2, 0, 1, 32)
    canon = Shapes.of(np.int64(3), np.int32(2), 1, 64)  # type: ignore[arg-type]
    assert type(canon.n) is int and canon == Shapes.of(3, 2, 1, 64)
    assert repr(canon) == repr(Shapes.of(3, 2, 1, 64))


def test_bucket():
    assert [bucket(t, 32) for t in (1, 20, 32, 33, 50, 64, 300)] == [32, 32, 32, 64, 64, 64, 320]
    assert bucket(7, 1) == 7 and bucket(np.int64(33), 32) == 64
    for bad in ((0, 32), (5, 0), (5.0, 32), (True, 32)):
        with pytest.raises(ValueError):
            bucket(*bad)


# ------------------------------------------------------------------------------ dataclasses


def test_static_config_is_a_value():
    a, b = make(), make()
    assert a == b and hash(a) == hash(b) and a is not b
    calls = []

    @functools.cache
    def factory(static: StaticConfig) -> int:
        calls.append(static)
        return len(calls)

    assert factory(a) == factory(b) == 1
    variants = [
        make(shapes=Shapes.of(3, 2, 1, 96)),
        make(vol=VolStatic("sv", exact=False)),
        make(vol=VolStatic("sv", nu_grid=(3, 4))),
        make(mean=MeanStatic("linear_structural", gram="batched")),
        make(store=StoreStatic(paths=True)),
        make(batching="map"),
        make(chunk=50),
        make(float32_leaves={"h"}),
    ]
    assert len({a, *variants}) == len(variants) + 1
    assert make(float32_leaves=["h"]).float32_leaves == frozenset({"h"})


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StructStatic(kind="factor"), "StructStatic.kind"),
        (lambda: StructStatic(contemp_scaling="sigma"), "contemp_scaling"),
        (lambda: MeanStatic("bart"), "MeanStatic.kind"),
        (lambda: MeanStatic("linear_structural", const_path="auto"), "const_path"),
        (lambda: MeanStatic("linear_structural", gram="auto"), "gram"),
        (lambda: MeanStatic("linear_structural", gram="packed_tiled"), "gram_tile"),
        (lambda: MeanStatic("linear_structural", gram_tile=512), "gram_tile"),
        (lambda: MeanStatic("linear_structural", cccm_precompute=1), "cccm_precompute"),
        (lambda: CoefPriorStatic(kind="cauchy_prior"), "CoefPriorStatic.kind"),
        (lambda: CoefPriorStatic(kind="horseshoe"), "n_groups"),  # Phase 3: G must be resolved
        (lambda: VolStatic("garch"), "persistent"),
        (lambda: VolStatic("sv", transitory="cauchy"), "transitory"),
        (lambda: VolStatic("sv", mixture="ksc10"), "mixture"),
        (lambda: VolStatic("sv", h0="diffuse"), "h0"),
        (lambda: VolStatic("sv", sigma2_prior="gamma"), "sigma2_prior"),
        (lambda: VolStatic("sv", t_scale="raw"), "t_scale"),
        (lambda: VolStatic("sv", exact=1), "exact"),
        (lambda: VolStatic("sv", nu_grid=()), "nu_grid"),
        (lambda: VolStatic("sv", nu_grid=(5, 4)), "nu_grid"),
        (lambda: VolStatic("sv", nu_grid=(0, 4)), "nu_grid"),
        (lambda: VolStatic("sv", nu_grid=(3.5,)), "nu_grid"),
        (lambda: StoreStatic(in_loop_forecast=-1), "in_loop_forecast"),
        (lambda: make(batching="pmap"), "batching"),
        (lambda: make(chunk=0), "chunk"),
        (lambda: make(float32_leaves="h"), "float32_leaves"),
        (lambda: make(vol="sv"), "StaticConfig.vol"),
    ],
)
def test_field_validation(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_vol_static_normalises_and_classifies():
    v = VolStatic("sv", transitory="outlier_t", nu_grid=[np.int64(5), 7, 9])  # type: ignore[arg-type]
    assert v.nu_grid == (5, 7, 9) and all(type(x) is int for x in v.nu_grid)
    assert hash(v) == hash(VolStatic("sv", transitory="outlier_t", nu_grid=(5, 7, 9)))
    assert v.has_outliers and v.has_t and not v.constant_weights
    assert VolStatic("constant").constant_weights
    assert not VolStatic("constant", transitory="outlier").constant_weights
    assert VolStatic("sv").nu_grid == tuple(range(3, 41))


# ------------------------------------------------------------------- composition rule table

S0, SL = StructStatic(), StructStatic(contemp_scaling="lambda")
LIN, RED, ZERO = "linear_structural", "linear_reduced", "zero"
CONST, CONST_C = VolStatic("constant"), VolStatic("constant", coupled=True)
SV = VolStatic("sv")

LEGAL = {
    "mode1_sv": (S0, MeanStatic(LIN), SV),
    "mode1_constant_eigen": (S0, MeanStatic(LIN, const_path="eigen"), CONST),
    "mode1_constant_cholesky": (S0, MeanStatic(LIN, const_path="cholesky_const"), CONST),
    "mode1_constant_weighted": (S0, MeanStatic(LIN), CONST),
    "mode2_fixed_factor": (SL, MeanStatic(LIN, "lambda", "fixed_factor"), CONST_C),
    "mode2_cholesky": (SL, MeanStatic(LIN, "lambda", "cholesky_const"), CONST_C),
    "mode2_weighted": (SL, MeanStatic(LIN, "lambda"), CONST_C),
    "iw_equivalent_structural": (SL, MeanStatic(LIN, const_path="cholesky_const"), CONST_C),
    "reduced_sv": (S0, MeanStatic(RED), VolStatic("sv", transitory="outlier_t")),
    "reduced_constant": (S0, MeanStatic(RED, cccm_precompute=False), CONST),
    "reduced_iw_equivalent": (SL, MeanStatic(RED), CONST_C),
    "zero_sv": (S0, MeanStatic(ZERO), VolStatic("sv", zero_offset=True)),
    "rw_ignores_h0": (S0, MeanStatic(LIN), VolStatic("sv", rw=True, h0="fixed_b0")),
    "t_unit_scale_nu_2": (
        S0,
        MeanStatic(LIN),
        VolStatic("sv", transitory="t", t_scale="unit", nu_grid=(1, 2)),
    ),
    "outlier_K_unused": (S0, MeanStatic(LIN), VolStatic("sv", outlier_K=1)),
    "standardized_grid_unused": (
        S0,
        MeanStatic(LIN),
        VolStatic("sv", transitory="outlier", nu_grid=(2, 3)),
    ),
    "constant_transitory": (S0, MeanStatic(LIN), VolStatic("constant", transitory="t")),
    "mode2_transitory": (
        SL,
        MeanStatic(LIN, "lambda"),
        VolStatic("constant", transitory="outlier", coupled=True),
    ),
    "tiled_gram": (S0, MeanStatic(LIN, gram="packed_tiled", gram_tile=512), SV),
}

ILLEGAL = {
    "mode2_needs_contemp_scaling": (S0, MeanStatic(LIN, "lambda"), CONST_C, "requires contemp_scaling"),
    "mode2_reduced": (SL, MeanStatic(RED, "lambda"), CONST_C, "only for the structural"),
    "mode2_zero": (SL, MeanStatic(ZERO, "lambda"), CONST_C, "only for the structural"),
    "scaling_with_sv": (SL, MeanStatic(LIN), VolStatic("sv"), "coupled=True"),
    "scaling_uncoupled": (SL, MeanStatic(LIN), CONST, "coupled=True"),
    "coupled_sv_core": (S0, MeanStatic(LIN), VolStatic("sv", coupled=True), "Constant persistent core"),
    "coupled_without_scaling": (S0, MeanStatic(LIN), CONST_C, "without a λ-scaled"),
    "fast_path_reduced": (S0, MeanStatic(RED, const_path="cholesky_const"), CONST, "structural Linear path"),
    "fast_path_zero": (S0, MeanStatic(ZERO, const_path="eigen"), CONST, "structural Linear path"),
    "eigen_with_sv": (S0, MeanStatic(LIN, const_path="eigen"), SV, "time-constant weights"),
    "eigen_with_transitory": (S0, MeanStatic(LIN, const_path="eigen"), VolStatic("constant", transitory="outlier"), "time-constant weights"),
    "cholesky_const_with_sv": (S0, MeanStatic(LIN, const_path="cholesky_const"), SV, "time-constant weights"),
    "fixed_factor_with_transitory": (SL, MeanStatic(LIN, "lambda", "fixed_factor"), VolStatic("constant", transitory="t", coupled=True), "time-constant weights"),
    "eigen_mode2": (SL, MeanStatic(LIN, "lambda", "eigen"), CONST_C, "eigen' needs Mode I"),
    "eigen_iw_equivalent": (SL, MeanStatic(LIN, const_path="eigen"), CONST_C, "eigen' needs Mode I"),
    "fixed_factor_mode1": (S0, MeanStatic(LIN, const_path="fixed_factor"), CONST, "fixed_factor' needs Mode II"),
    "fixed_factor_iw_equivalent": (SL, MeanStatic(LIN, const_path="fixed_factor"), CONST_C, "fixed_factor' needs Mode II"),
    "rw_constant": (S0, MeanStatic(LIN), VolStatic("constant", rw=True), "rw=True"),
    "zero_offset_constant": (S0, MeanStatic(LIN), VolStatic("constant", zero_offset=True), "zero_offset"),
    "outlier_K_1": (S0, MeanStatic(LIN), VolStatic("sv", transitory="outlier", outlier_K=1), "outlier_K"),
    "outlier_t_K_1": (S0, MeanStatic(LIN), VolStatic("sv", transitory="outlier_t", outlier_K=1), "outlier_K"),
    "standardized_nu_2": (S0, MeanStatic(LIN), VolStatic("sv", transitory="t", nu_grid=(2, 3, 4)), "ν > 2"),
    "standardized_outlier_t_nu_1": (S0, MeanStatic(LIN), VolStatic("sv", transitory="outlier_t", nu_grid=(1, 5)), "ν > 2"),
}  # fmt: skip


@pytest.mark.parametrize("name", list(LEGAL))
def test_legal_compositions(name):
    struct, mean, vol = LEGAL[name]
    validate_composition(struct, mean, CoefPriorStatic(), vol)
    make(struct=struct, mean=mean, vol=vol)  # construction validates too


@pytest.mark.parametrize("name", list(ILLEGAL))
def test_illegal_compositions(name):
    struct, mean, vol, message = ILLEGAL[name]
    with pytest.raises(ValueError, match=message):
        validate_composition(struct, mean, CoefPriorStatic(), vol)
    with pytest.raises(ValueError, match=message):
        make(struct=struct, mean=mean, vol=vol)


def test_capability_tables_and_varying_coefficient_priors(monkeypatch):
    assert cfg.SUPPORTS_REDUCED_MEAN == {"recursive": True}
    assert cfg.COEF_VARIES == {  # Phase 3 adds the global-local and SSVS priors (P3 D-43)
        "fixed_diagonal": False,
        **dict.fromkeys(("horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"), True),
    }
    monkeypatch.setitem(cfg.SUPPORTS_REDUCED_MEAN, "recursive", False)
    with pytest.raises(ValueError, match="does not support a reduced-form mean"):
        validate_composition(S0, MeanStatic(RED), CoefPriorStatic(), SV)
    monkeypatch.setitem(cfg.SUPPORTS_REDUCED_MEAN, "recursive", True)
    monkeypatch.setitem(cfg.COEF_VARIES, "fixed_diagonal", True)  # a Phase 3 global-local prior
    with pytest.raises(ValueError, match="non-varying coefficient prior"):
        validate_composition(S0, MeanStatic(LIN, const_path="eigen"), CoefPriorStatic(), CONST)
    with pytest.raises(ValueError, match="non-varying coefficient prior"):
        validate_composition(
            SL, MeanStatic(LIN, "lambda", "fixed_factor"), CoefPriorStatic(), CONST_C
        )
    # Phase 3 rule (11): a varying prior needs the weighted path (P3 §2.1), so the Phase 2
    # expectation "cholesky_const stays legal" no longer holds
    with pytest.raises(ValueError, match="needs a fixed prior"):
        validate_composition(
            SL,
            MeanStatic(LIN, "lambda", "cholesky_const", coef_log=True),
            CoefPriorStatic(),
            CONST_C,
        )


def test_scaling_mode():
    assert scaling_mode(S0, MeanStatic(LIN)) == "I"
    assert scaling_mode(SL, MeanStatic(LIN, "lambda")) == "II"
    assert scaling_mode(SL, MeanStatic(RED)) == "iw_equivalent"


# ---------------------------------------------------------------------------------- registry


def test_registry_covers_every_static_kind():
    kinds = set(registry.KERNELS)
    for alias in (cfg.StructKind, cfg.MeanKind, cfg.CoefKind, cfg.Persistent, cfg.Transitory):
        assert set(get_args(alias)) <= kinds
    assert {"cccm", "vol_block"} <= kinds
    assert set(registry.OWNERS) == kinds
    for kind, target in registry.KERNELS.items():
        module, _, cls = target.partition(":")
        assert module.startswith("rai_bvar.components.") and cls.isidentifier(), kind


def test_slot_kinds():
    st = make(mean=MeanStatic(RED), vol=VolStatic("sv", transitory="t"))
    assert registry.slot_kinds(st) == {
        "structure": "recursive",
        "mean": "linear_reduced",
        "coef": "fixed_diagonal",
        "core": "sv",
        "transitory": "t",
        "vol": "vol_block",
        "global_pre": ("cccm",),
        "global_vol": (),
    }
    assert registry.slot_kinds(make())["global_pre"] == ()
    common = make(vol=VolStatic("constant", common=True))
    assert registry.slot_kinds(common)["global_vol"] == ("common_sv",)


def test_load(monkeypatch, tmp_path):
    with pytest.raises(KeyError, match="unknown kernel kind"):
        registry.load("garch")
    monkeypatch.setitem(registry.KERNELS, "toy", "rai_bvar.engines.mcmc.config:Shapes")
    assert registry.load("toy") is Shapes
    monkeypatch.setitem(registry.KERNELS, "toy", "rai_bvar.engines.mcmc.config:NoSuchClass")
    with pytest.raises(NotImplementedError, match="defines no NoSuchClass"):
        registry.load("toy")
    monkeypatch.setitem(registry.KERNELS, "toy", "rai_bvar.engines.mcmc.config:bucket")
    with pytest.raises(TypeError, match="not a class"):
        registry.load("toy")
    monkeypatch.setitem(registry.KERNELS, "toy", "rai_bvar.components.vol.not_written_yet:X")
    with pytest.raises(NotImplementedError, match="cannot be imported"):
        registry.load("toy")
    # an import error *inside* an existing kernel module is not masked
    (tmp_path / "wp1_broken_kernel.py").write_text("import wp1_missing_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry.KERNELS, "toy", "wp1_broken_kernel:X")
    with pytest.raises(ModuleNotFoundError, match="wp1_missing_dependency_xyz"):
        registry.load("toy")
