"""Phase 4 week-1 spike: the bartz adapter and the BART reference model (SPEC §13 Phase 4;
``docs/phase4/SPIKE.md``). Skipped unless the optional ``bart`` extra (``bartz==0.13.0``) is
installed (``uv sync --group dev --extra bart``); CI installs the dev group only.

Question 1 (the setter): :func:`set_data` on a root-only state equals ``mcmcstep.init`` on the
same data in every field it writes, and one ``step`` from either is bitwise identical; on a grown
forest it reproduces the caches bartz maintains itself. Question 3 (fixed error precision): with
``Wishart(None, None, 1)`` the σ update is skipped and the per-observation weights are the exact
leaf-likelihood precisions; ``init`` refuses that hook for continuous outcomes. Question 2's
precondition: bartz 0.13.0 does not trace under x64 unpatched, and does with the two-line patch.
The reference tree prior (two-implementation rule) is checked against ``bartz.debug``'s sampler
and against exact enumeration.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

bartz = pytest.importorskip("bartz", reason="optional extra 'bart' (bartz) not installed")

from bartz import mcmcstep  # noqa: E402
from bartz.grove import is_actual_leaf  # noqa: E402
from bartz.mcmcstep._step import apply_moves_to_leaf_indices  # noqa: E402

from rai_bvar.components.mean import _bart_spike as bs  # noqa: E402
from rai_bvar.engines.mcmc.state import assert_float64  # noqa: E402
from rai_bvar.validation import _bart_spike as ref  # noqa: E402

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

PRIOR = bs.BartPrior(num_trees=20, max_depth=5)
T, PX = 64, 4


def _data(seed: int = 0, T_eff: int = 50):
    """Bins, an f32-representable target and power-of-4 weights (so ``1/error_scale`` is exact
    in float32 and init's and the adapter's derivations can agree bitwise); padded rows weight 0."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, PRIOR.max_split + 1, size=(PX, T)).astype(np.uint8)
    y = rng.normal(size=T).astype(np.float32).astype(np.float64)
    k = rng.integers(-3, 4, size=T)
    w = np.where(np.arange(T) < T_eff, 4.0**k, 0.0)
    return jnp.asarray(X), jnp.asarray(y), jnp.asarray(w)


def _init_like_template(X, y, w, prior=PRIOR):
    """``mcmcstep.init`` on ``(X, y, error_scale = w^{−½})`` with padded rows ``missing``, the
    fixed error precision swapped in and the dtypes of :func:`template_state`."""
    live = np.asarray(w) > 0
    scale = np.where(live, 1.0 / np.sqrt(np.where(live, np.asarray(w), 1.0)), 1.0)
    with jax.enable_x64(False):
        st = mcmcstep.init(
            X=np.asarray(X),
            y=np.asarray(y, np.float32),
            offset=0.0,
            max_split=np.full(PX, prior.max_split, np.uint8),
            num_trees=prior.num_trees,
            p_nonterminal=prior.p_nonterminal(),
            leaf_prior_cov_inv=mcmcstep.Wishart(None, None, value=1.0 / prior.leaf_sd**2),
            error_cov_inv=mcmcstep.Wishart(nu=1.0, rate=1.0, value=1.0),
            error_scale=scale.astype(np.float32),
            missing=~live,
        )
        st = dataclasses.replace(st, error_cov_inv=mcmcstep.Wishart(None, None, value=1.0))
    tmpl = bs.template_state(prior, PX, T)
    return jax.tree.map(lambda a, t: jnp.asarray(a, t.dtype), st, tmpl)


def _get(st, path: str):
    for part in path.split("."):
        st = getattr(st, part)
    return st


@jax.jit
def _step(key, st):
    return bs.bartz_step(key, st, False)


def _leaves(tree):
    return jax.tree_util.tree_flatten_with_path(tree)[0]


# --------------------------------------------------------------------------- question 1: setter


def test_set_data_on_a_stump_equals_init_bitwise():
    """Every field the setter writes equals ``init``'s value, and one ``step`` from the two states
    (same key) is bitwise identical in every leaf."""
    X, y, w = _data()
    mine = jax.jit(bs.set_data)(jax.tree.map(jnp.asarray, bs.template_state(PRIOR, PX, T)), X, y, w)
    theirs = _init_like_template(X, y, w)
    for path in bs.SET_FIELDS:
        a, b = np.asarray(_get(mine, path)), np.asarray(_get(theirs, path))
        assert a.dtype == b.dtype and np.array_equal(a, b), path
    key = random.key(7)
    out_a, out_b = _step(key, mine), _step(key, theirs)
    for (p, a), (_, b) in zip(_leaves(out_a), _leaves(out_b), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b)), jax.tree_util.keystr(p)


def test_set_data_reproduces_bartz_caches_on_a_grown_forest():
    """After 100 steps, ``set_data`` with the same data reproduces bartz's own caches: leaf indices
    equal after folding the pending prunes, ``prec_tree`` at the actual leaves to float32
    summation order, residuals to float32 running-update error."""
    X, y, w = _data(1)
    st = jax.tree.map(jnp.asarray, bs.template_state(PRIOR, PX, T))

    @jax.jit
    def run(key, s):
        def body(s, k):
            return _step(k, bs.set_data(s, X, y, w)), None

        return jax.lax.scan(body, s, random.split(key, 100))[0]

    s1 = _step(random.key(3), bs.set_data(run(random.key(2), st), X, y, w))
    s2 = jax.jit(bs.set_data)(s1, X, y, w)
    assert int(jnp.sum(s1.forest.split_tree > 0)) > PRIOR.num_trees // 2  # grown, not stumps
    folded = apply_moves_to_leaf_indices(
        s1.forest.leaf_indices, s1.forest.to_prune, s1.forest.move_node
    )
    assert np.array_equal(np.asarray(folded), np.asarray(s2.forest.leaf_indices))
    leaf = jax.vmap(lambda t: is_actual_leaf(t, add_bottom_level=True))(s1.forest.split_tree)
    a, b = np.asarray(s1.forest.prec_tree), np.asarray(s2.forest.prec_tree)
    np.testing.assert_allclose(a[np.asarray(leaf)], b[np.asarray(leaf)], rtol=2e-6, atol=0)
    live = np.asarray(w) > 0
    np.testing.assert_allclose(np.asarray(s1.resid)[live], np.asarray(s2.resid)[live], atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(bs.forest_values(s1.forest, s1.forest.leaf_indices))[live],
        np.asarray(bs.forest_values(s2.forest, s2.forest.leaf_indices))[live],
        rtol=0, atol=1e-12,
    )  # fmt: skip


def test_swapping_weights_changes_only_the_written_fields():
    """The setter touches exactly :data:`SET_FIELDS` (plus nothing else): a regression guard for
    bartz releases that add derived fields."""
    X, y, w = _data(2)
    X2, y2, w2 = _data(3)
    st = jax.jit(bs.set_data)(jax.tree.map(jnp.asarray, bs.template_state(PRIOR, PX, T)), X, y, w)
    st2 = jax.jit(bs.set_data)(st, X2, y2, w2)
    changed = {
        jax.tree_util.keystr(p, simple=True, separator=".")
        for (p, a), (_, b) in zip(_leaves(st), _leaves(st2), strict=True)
        if not np.array_equal(np.asarray(a), np.asarray(b))
    }
    assert changed <= set(bs.SET_FIELDS) and changed >= {"X", "resid", "prec_scale"}


# --------------------------------------------------------------------------- question 3: precision


def test_init_rejects_a_prior_free_error_precision():
    """The documented hook (``Wishart`` with ``nu = rate = None``: precision held fixed) is refused
    by ``init`` for continuous outcomes; the adapter swaps it in after ``init``."""
    with jax.enable_x64(False), pytest.raises(AssertionError):
        mcmcstep.init(
            X=np.zeros((1, 8), np.uint8), y=np.zeros(8, np.float32), offset=0.0,
            max_split=np.full(1, 3, np.uint8), num_trees=2,
            p_nonterminal=PRIOR.p_nonterminal(),
            leaf_prior_cov_inv=mcmcstep.Wishart(None, None, value=1.0),
            error_cov_inv=mcmcstep.Wishart(None, None, value=1.0),
        )  # fmt: skip


def test_fixed_precision_and_weights_are_exact_leaf_precisions():
    """One tree held at the root (``min_points_per_leaf > T``): every ``step`` redraws the root leaf
    from ``N(Σw·y/(p₀ + Σw), 1/(p₀ + Σw))``, with ``w = 1/error_scale²`` and ``p₀`` the leaf prior
    precision, and leaves ``error_cov_inv = 1`` untouched (no σ update). 4000 draws: z-tests on
    the mean and on the variance."""
    rng = np.random.default_rng(5)
    n = 40
    y = rng.normal(1.0, 1.0, n).astype(np.float32)
    w = np.exp(rng.uniform(-3.0, 3.0, n))  # a volatility range of e⁶ ≈ 400
    p0 = 4.0
    with jax.enable_x64(False):
        st = mcmcstep.init(
            X=rng.integers(0, 8, size=(2, n)).astype(np.uint8), y=y, offset=0.0,
            max_split=np.full(2, 7, np.uint8), num_trees=1, p_nonterminal=PRIOR.p_nonterminal(),
            leaf_prior_cov_inv=mcmcstep.Wishart(None, None, value=p0),
            error_cov_inv=mcmcstep.Wishart(nu=1.0, rate=1.0, value=1.0),
            error_scale=(1.0 / np.sqrt(w)).astype(np.float32), min_points_per_leaf=n + 1,
        )  # fmt: skip
        st = dataclasses.replace(st, error_cov_inv=mcmcstep.Wishart(None, None, value=1.0))

        @jax.jit
        def draws(keys):
            def one(k):
                s = mcmcstep.step(k, st)
                leaf = s.forest.leaf_tree[0, 1] * s.forest.leaf_unit
                return leaf, s.error_cov_inv.value, s.forest.split_tree[0, 1]

            return jax.vmap(one)(keys)

        leaf, ecov, split = draws(random.split(random.key(0), 4000))
    w32 = (1.0 / np.asarray((1.0 / np.sqrt(w)).astype(np.float32), np.float64)) ** 2
    prec = p0 + w32.sum()
    mean = (w32 * y.astype(np.float64)).sum() / prec
    leaf = np.asarray(leaf, np.float64)
    assert np.all(np.asarray(ecov) == 1.0) and np.all(np.asarray(split) == 0)
    assert abs(leaf.mean() - mean) / math.sqrt(1.0 / prec / leaf.size) < 4.0
    var_z = (leaf.var(ddof=1) * prec - 1.0) / math.sqrt(2.0 / (leaf.size - 1))
    assert abs(var_z) < 4.0


# --------------------------------------------------------------------------- x64


def _x64_state():
    with jax.enable_x64(True):
        st = mcmcstep.init(
            X=np.zeros((2, 16), np.uint8), y=np.zeros(16, np.float32), offset=0.0,
            max_split=np.full(2, 7, np.uint8), num_trees=4, p_nonterminal=PRIOR.p_nonterminal(),
            leaf_prior_cov_inv=mcmcstep.Wishart(None, None, value=1.0),
            error_cov_inv=mcmcstep.Wishart(nu=1.0, rate=1.0, value=1.0),
            error_scale=np.ones(16, np.float32), resid_dtype=jnp.float64,
            prec_scale_dtype=jnp.float64,
        )  # fmt: skip
    return dataclasses.replace(st, error_cov_inv=mcmcstep.Wishart(None, None, value=1.0))


@pytest.mark.filterwarnings("ignore:scatter inputs have incompatible types:FutureWarning")
def test_bartz_step_does_not_trace_under_x64():
    """bartz 0.13.0 under ``jax_enable_x64``: ``lrt_nodes`` is int64 and ``step`` asserts int32."""
    st = _x64_state()
    with pytest.raises(AssertionError):
        jax.eval_shape(mcmcstep.step, random.key(0), st)


@pytest.mark.filterwarnings("error::FutureWarning")
def test_two_line_patch_traces_under_x64_with_float64_residuals():
    """With :func:`bartz_x64_patch` the step traces under x64, keeps every dtype (float64
    residuals and precision scales, float32 leaves) and raises no dtype warning."""
    st = _x64_state()
    with ref.bartz_x64_patch():
        out = jax.eval_shape(mcmcstep.step, random.key(0), st)
    for (p, a), (_, b) in zip(_leaves(st), _leaves(out), strict=True):
        assert a.dtype == b.dtype, jax.tree_util.keystr(p)
    assert st.resid.dtype == jnp.float64 and st.forest.leaf_tree.dtype == jnp.float32


def test_float32_leaves_are_the_bartz_state_only():
    """Path (a): every float leaf of the bartz sub-state is float32 (and declared); the rest of the
    chain state stays float64 (R4)."""
    paths = bs.float_leaf_paths(PRIOR, PX, T)
    assert {"eq.mean.resid", "eq.mean.prec_scale", "eq.mean.forest.leaf_tree"} <= paths
    J = ref.bart_joint("bart_sv_dtypes", bart=PRIOR, n=2, T_eff=50)
    st = jax.jit(J.prior_sample)(random.key(0))
    assert_float64(st, allow=bs.float_leaf_paths(PRIOR, J.p_x, J.static.shapes.T_pad))
    with pytest.raises(TypeError):
        assert_float64(st)


# --------------------------------------------------------------------------- reference model


def test_reference_evaluator_matches_bartz():
    """Box membership (reference) and heap descent (kernel ``traverse``; bartz's
    ``grove.evaluate_forest``) give the same ``g`` on prior forests."""
    from bartz.grove import TreesTrace, evaluate_forest

    X, _, _ = _data(4)
    for seed in range(5):
        forest, _ = jax.jit(lambda k: ref.forest_prior_sample(k, PRIOR, PX))(random.key(seed))
        g_ref = ref.forest_eval(forest, X.astype(jnp.int32), PRIOR.max_split)
        idx = bs.traverse(X, forest.var, forest.split, jnp.uint8)
        g_kernel = jnp.sum(jnp.take_along_axis(forest.leaf, idx.astype(jnp.int32), -1), axis=0)
        trees = TreesTrace(
            var_tree=forest.var.astype(jnp.uint8), split_tree=forest.split.astype(jnp.uint8),
            leaf_tree=forest.leaf.astype(jnp.float32),
        )  # fmt: skip
        g_bartz = evaluate_forest(X, trees, sum_batch_axis=0)
        np.testing.assert_allclose(np.asarray(g_kernel), np.asarray(g_ref), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(g_bartz), np.asarray(g_ref), rtol=1e-5, atol=1e-6)


def _summaries(var, split, half):
    var, split = np.asarray(var, np.int64), np.asarray(split, np.int64)
    internal = split[..., 1:half] > 0
    n_int = internal.sum(-1)
    root_var = np.where(split[..., 1] > 0, var[..., 1], -1)
    root_cut = np.where(split[..., 1] > 0, split[..., 1], 0)
    return n_int.ravel(), root_var.ravel(), root_cut.ravel()


def _chi2_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    table = np.array([[np.sum(a == c) for c in cats], [np.sum(b == c) for c in cats]])
    table = table[:, table.sum(0) >= 10]
    return stats.chi2_contingency(table)[1]


def test_reference_prior_sampler_matches_bartz_debug():
    """20,000 trees from the reference sampler and from ``bartz.debug.sample_prior``: the internal
    node counts, root split variables and root cutpoints agree (two-sample χ², p > 1e-3), and the
    leaves have s.d. ``leaf_sd``."""
    from bartz.debug import sample_prior

    prior = bs.BartPrior(num_trees=1, max_depth=5, alpha=0.95, beta=1.0, cuts=(-1.0, 0.0, 1.0, 2.0))
    K = 20000
    forest, _ = jax.jit(jax.vmap(lambda k: ref.forest_prior_sample(k, prior, 3)))(
        random.split(random.key(11), K)
    )
    with jax.enable_x64(False):
        theirs = sample_prior(
            random.key(12), K, 1, jnp.full(3, prior.max_split, jnp.uint8),
            jnp.asarray(prior.p_nonterminal()), jnp.float32(prior.leaf_sd),
        )  # fmt: skip
    half = 2 ** (prior.max_depth - 1)
    for a, b in zip(
        _summaries(forest.var, forest.split, half),
        _summaries(theirs.var_tree, theirs.split_tree, half),
        strict=True,
    ):
        assert _chi2_two_sample(a, b) > 1e-3
    sd = float(np.std(np.asarray(forest.leaf)))
    assert abs(sd / prior.leaf_sd - 1.0) < 0.02


def test_reference_tree_log_prior_matches_sampler_frequencies():
    """Depth 3, two variables, three cutpoints: the structure log-prior of every sampled tree
    equals the log of its sampling frequency (χ² goodness of fit), and the probabilities of the
    observed structures sum to ≈ 1."""
    prior = bs.BartPrior(num_trees=1, max_depth=3, alpha=0.9, beta=0.5, cuts=(-1.0, 0.0, 1.0))
    K = 60000
    forest, _ = jax.jit(jax.vmap(lambda k: ref.forest_prior_sample(k, prior, 2)))(
        random.split(random.key(21), K)
    )
    lp = jax.jit(jax.vmap(lambda f: ref.tree_log_prior(f, prior, 2, leaves=False)))(forest)
    var, split = np.asarray(forest.var)[:, 0], np.asarray(forest.split)[:, 0]
    keys = [tuple(np.where(s > 0, v + 1, 0)) + tuple(s) for v, s in zip(var, split, strict=True)]
    uniq, inv, counts = np.unique(np.array(keys), axis=0, return_inverse=True, return_counts=True)
    p = np.zeros(len(uniq))
    p[inv.ravel()] = np.exp(np.asarray(lp)[:, 0] if np.ndim(lp) > 1 else np.asarray(lp))
    assert abs(p.sum() - 1.0) < 0.01
    big = K * p >= 5
    obs = np.append(counts[big], counts[~big].sum())
    exp = np.append(K * p[big], K * (1.0 - p[big].sum()))
    assert stats.chisquare(obs, exp).pvalue > 1e-3


def test_joint_model_runs_through_the_production_sweep():
    """The spike composition's plan is ``mean → contemp → residual → vol`` and ten successive-
    conditional iterations keep every test function finite and the status clean."""
    J = ref.bart_joint("bart_sv_smoke", bart=PRIOR, n=2, T_eff=50)
    assert J.plan.names == ("mean", "contemp", "residual", "vol")
    key = random.key(0)

    @jax.jit
    def run(st):
        def body(s, it):
            d = J.simulate(random.fold_in(key, it), s)
            s = J.step(key, it, s, d)
            return s, J.test_functions(s, d)

        return jax.lax.scan(body, st, jnp.arange(10, dtype=jnp.uint32))

    st, g = run(jax.jit(J.prior_sample)(key))
    assert np.all(np.isfinite(np.asarray(g))) and int(st.status.code) == 0
    assert len(J.names) >= 60


def test_padded_rows_never_enter_the_leaf_likelihood():
    """Rows with zero weight drop out like bartz's ``missing``: changing their targets and bins
    leaves every cache that reaches the likelihood unchanged."""
    X, y, w = _data(6, T_eff=40)
    pad = np.asarray(w) == 0
    X2 = jnp.where(jnp.asarray(pad)[None, :], 0, X).astype(jnp.uint8)
    y2 = jnp.where(jnp.asarray(pad), 1e6, y)
    tmpl = jax.tree.map(jnp.asarray, bs.template_state(PRIOR, PX, T))
    a = jax.jit(bs.set_data)(tmpl, X, y, w)
    b = jax.jit(bs.set_data)(tmpl, X2, y2, w)
    for f in (
        "prec_scale",
        "inv_sdev_scale",
        "inv_sdev_unit",
        "n_non_missing",
        "sum_diag_prec_scale",
    ):
        assert np.array_equal(np.asarray(_get(a, f)), np.asarray(_get(b, f))), f
    np.testing.assert_array_equal(np.asarray(a.forest.prec_tree), np.asarray(b.forest.prec_tree))
    assert int(a.n_non_missing) == 40
