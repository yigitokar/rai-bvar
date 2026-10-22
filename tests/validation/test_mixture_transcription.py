"""Second transcription of the OCSN-10 / KSC-7 mixture tables (DESIGN §9.1; N01 §5, §12.1).

The reference tables in ``validation/reference.py`` must agree on every printed digit with
(i) the tables of research note 01 (parsed from the Markdown at test time), (ii) the derived
constants and moments printed there, and (iii) the kernel transcription in
``components/vol/mixture.py`` once WP12 provides it.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from rai_bvar.validation import reference as R

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

NOTE = Path(__file__).resolve().parents[2] / "docs" / "phase2" / "01_omori_mixture.md"
DIGITS = 5  # every source prints five decimals (N01 §5)


def _section(title_prefix: str) -> str:
    text = NOTE.read_text(encoding="utf-8")
    start = text.index(title_prefix)
    nxt = text.find("\n### ", start + len(title_prefix))
    return text[start : nxt if nxt != -1 else None]


def _rows(section: str, ncols: int) -> np.ndarray:
    """Numeric rows of the first Markdown table in ``section`` (columns after the index)."""
    out = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < ncols + 1 or not re.fullmatch(r"\d+", cells[0]):
            continue
        out.append([float(c.replace("−", "-")) for c in cells[1 : ncols + 1]])
    return np.array(out)


def _printed(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64), DIGITS)


def test_ocsn10_matches_note_01_section_5_3():
    note = _rows(_section("### 5.3 OCSN 10-component"), 3)  # p, m, v
    assert note.shape == (10, 3)
    tab = R.mixture_table("ocsn10")
    np.testing.assert_array_equal(_printed(tab.p), note[:, 0])
    np.testing.assert_array_equal(_printed(tab.m), note[:, 1])  # no shift (D-3)
    np.testing.assert_array_equal(_printed(tab.v), note[:, 2])


def test_ksc7_raw_and_effective_match_note_01_sections_5_1_and_5_2():
    raw = _rows(_section("### 5.1 KSC 7-component"), 4)  # p, m (centred), v, effective mean
    np.testing.assert_array_equal(np.array(R._KSC7_TABLE4_PRINTED), raw[:, :3])
    np.testing.assert_array_equal(_printed(raw[:, 1] - R.KSC_SHIFT), raw[:, 3])
    shifted = _rows(_section("### 5.2 KSC 7-component as reprinted"), 3)  # OCSN left block
    tab = R.mixture_table("ksc7")
    np.testing.assert_array_equal(_printed(tab.p), shifted[:, 0])
    np.testing.assert_array_equal(_printed(tab.m), shifted[:, 1])
    np.testing.assert_array_equal(_printed(tab.v), shifted[:, 2])


def test_both_tables_sorted_by_descending_effective_mean():
    for name in ("ocsn10", "ksc7"):
        assert np.all(np.diff(R.mixture_table(name).m) < 0), name


def test_unnormalised_log_weights_match_note_01_section_5_4():
    # κ_j = log p_j − ½ log v_j (N01 §5.4; stochvol stores the same numbers)
    ocsn = [-4.00937239, -2.17845316, -1.37686428, -1.12572770, -1.24873234, -1.66194609,
            -2.34338373, -3.35107342, -4.86438228, -7.76421433]  # fmt: skip
    ksc_table4 = [-5.79846149, -2.72885728, -11.64213255, -2.23086873, -0.85570701, -0.86474010,
                  -1.47332606]  # fmt: skip
    tab = R.mixture_table("ocsn10")
    np.testing.assert_allclose(np.log(tab.p) - 0.5 * np.log(tab.v), ocsn, atol=5e-9)
    raw = np.array(R._KSC7_TABLE4_PRINTED)
    np.testing.assert_allclose(np.log(raw[:, 0]) - 0.5 * np.log(raw[:, 2]), ksc_table4, atol=5e-9)


@pytest.mark.parametrize(
    ("name", "mean", "var"), [("ocsn10", -1.270280, 4.933731), ("ksc7", -1.270399, 4.934854)]
)
def test_mixture_moments_match_note_01(name, mean, var):
    tab = R.mixture_table(name)
    assert abs(tab.p.sum() - 1.0) < 1e-12 + 5e-6  # Σ p = 1.00000 as printed
    mu = float(np.sum(tab.p * tab.m))
    second = float(np.sum(tab.p * (tab.v + tab.m**2)))
    assert mu == pytest.approx(mean, abs=6e-7)
    assert second - mu**2 == pytest.approx(var, abs=6e-7)
    # exact log χ²₁: E ξ = ψ(½) + log 2 = −1.2703628…; the mixture is within 1e-4 (DESIGN §9.3 item 3)
    assert abs(mu - (special.digamma(0.5) + math.log(2.0))) < 1e-4


def test_log_densities_against_scipy():
    x = np.linspace(-20.0, 4.0, 97)
    for name in ("ocsn10", "ksc7"):
        tab = R.mixture_table(name)
        ref = np.log(np.sum(tab.p * stats.norm.pdf(x[:, None], tab.m, np.sqrt(tab.v)), axis=1))
        np.testing.assert_allclose(np.asarray(R.log_mixture_density(jnp.asarray(x), tab)), ref,
                                   rtol=1e-12, atol=1e-12)  # fmt: skip
        comps = np.asarray(R.log_mixture_components(jnp.asarray(x), tab))
        np.testing.assert_allclose(
            comps, np.log(tab.p) + stats.norm.logpdf(x[:, None], tab.m, np.sqrt(tab.v)), rtol=1e-12
        )
    # exact density: ξ = log ε², ε ~ N(0, 1)
    f = np.asarray(R.log_logchi2_density(jnp.asarray(x)))
    np.testing.assert_allclose(f, stats.chi2.logpdf(np.exp(x), 1) + x, rtol=1e-12, atol=1e-12)
    total, _ = integrate.quad(lambda u: math.exp(float(R.log_logchi2_density(u))), -60, 10)
    assert total == pytest.approx(1.0, abs=1e-10)


def _kernel_tables():
    try:
        mod = importlib.import_module("rai_bvar.components.vol.mixture")
    except Exception as err:  # pragma: no cover - depends on WP12 progress
        pytest.skip(f"components/vol/mixture.py not importable yet (WP12): {err}")
    if not all(hasattr(mod, a) for a in ("OCSN10", "KSC7")):
        pytest.skip("components/vol/mixture.py does not define OCSN10/KSC7 yet (WP12)")
    return {"ocsn10": mod.OCSN10, "ksc7": mod.KSC7}


@pytest.mark.parametrize("name", ["ocsn10", "ksc7"])
def test_kernel_transcription_agrees_on_every_printed_digit(name):
    kernel = _kernel_tables()[name]
    p, m, v = (np.asarray(getattr(kernel, a), dtype=np.float64) for a in ("p", "m", "v"))
    order = np.argsort(-m, kind="stable")  # compare as sets of rows (the kernel order is its own)
    ref = R.mixture_table(name)
    np.testing.assert_array_equal(_printed(p[order]), _printed(ref.p))
    np.testing.assert_array_equal(_printed(m[order]), _printed(ref.m))
    np.testing.assert_array_equal(_printed(v[order]), _printed(ref.v))


def test_unknown_mixture_raises():
    with pytest.raises(ValueError, match="unknown mixture"):
        R.mixture_table("ksc10")  # type: ignore[arg-type]
