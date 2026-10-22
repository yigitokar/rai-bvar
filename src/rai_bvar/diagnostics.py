"""MCMC convergence diagnostics (Vehtari et al. 2021): rank-normalised split R-hat and bulk ESS.

Self-contained NumPy implementations so the analytic engine's hyperparameter MH chains can be
checked without an ArviZ dependency.

Phase 2 adds (DESIGN §8.4, D-24) tail ESS, mean ESS and the MCSE of means and quantiles, batched
versions over the ``(chains, draws, *S)`` monitor arrays of the MCMC engine (``*_many``) and a
one-pass :func:`summary_many`; see the section "Phase 2 additions" below.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _split(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=np.float64)
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, chains.shape[1] - half :]], axis=0)


def _rank_normalise(chains: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(chains, method="average").reshape(chains.shape)
    return stats.norm.ppf((ranks - 0.375) / (chains.size + 0.25))


def _rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chains.mean(axis=1).var(ddof=1)
    if within <= 0:  # chains that never move are not evidence of convergence
        return np.nan
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def rhat(chains: np.ndarray) -> float:
    """Rank-normalised split R-hat: the max of the bulk and the folded (tail) versions."""
    split = _split(chains)
    if split.shape[1] < 2:
        return np.nan
    bulk = _rhat(_rank_normalise(split))
    folded = _rhat(_rank_normalise(np.abs(split - np.median(split))))
    return np.nan if np.isnan(bulk) or np.isnan(folded) else float(max(bulk, folded))


def ess_bulk(chains: np.ndarray) -> float:
    """Bulk effective sample size of rank-normalised split chains.

    Geyer's initial positive and monotone sequences with the "improved estimate" term and the
    lag bound ``n − 3``, as in Stan and ArviZ (Vehtari et al. 2021). Constant chains give NaN.
    """
    z = _rank_normalise(_split(chains))
    m, n = z.shape
    if n < 5:  # split halves of ≤ 4 draws carry no usable autocorrelation information
        return np.nan
    centred = z - z.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :n] / n
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + (z.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if not var_plus > 0:
        return np.nan

    def rho(t: int) -> float:
        return 1.0 - (mean_var - acov[:, t].mean()) / var_plus

    rho_hat = np.zeros(n)
    rho_even, rho_odd = 1.0, rho(1)
    rho_hat[0], rho_hat[1] = rho_even, rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:  # initial positive sequence
        rho_even, rho_odd = rho(t + 1), rho(t + 2)
        if rho_even + rho_odd >= 0:
            rho_hat[t + 1], rho_hat[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:  # improved estimate
        rho_hat[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:  # initial monotone sequence
        if rho_hat[t + 1] + rho_hat[t + 2] > rho_hat[t - 1] + rho_hat[t]:
            rho_hat[t + 1] = rho_hat[t + 2] = (rho_hat[t - 1] + rho_hat[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * rho_hat[: max_t + 1].sum() + rho_hat[max_t + 1 : max_t + 2].sum()
    total = m * n
    return float(total / max(tau, 1.0 / np.log10(total)))


# --------------------------------------------------------------------------- Phase 2 additions
# Tail ESS, mean ESS, MCSE of the mean and of quantiles, vectorised versions over many monitored
# quantities, and a one-pass summary (DESIGN §8.4, D-24; Vehtari et al. 2021 §4.3–4.4, as in
# ArviZ and Stan's ``posterior``). The functions above are unchanged. Shared conventions:
#
# * chains are split in halves (the middle draw of an odd length is dropped), so every ESS sees
#   ``2·chains`` sequences of ``draws // 2`` draws, and ESS is capped at ``S log10 S`` as in Stan;
# * split halves of fewer than 5 draws, NaN draws and chains that never move give NaN ESS and MCSE
#   (ArviZ returns its cap or the number of draws instead): no movement is no evidence of mixing;
# * ``*_many`` take ``(chains, draws, *S)`` and return an array of shape ``S``; the scalar
#   functions take ``(chains, draws)`` (a 1-D input is one chain) and run ``*_many`` on one column.
#
# The batched kernels keep every reduced axis last and contiguous and finish each quantity with the
# same scalar Geyer recursion as ``ess_bulk``, so ``rhat_many``/``ess_bulk_many`` agree with
# ``rhat``/``ess_bulk`` to rounding, not bitwise: the last bits of the FFT autocovariances depend on
# the batch a quantity runs in (its size and neighbours), so a quantity's ESS can differ in the
# last digits between batches (DESIGN §8.2). Cost per quantity: two argsorts (bulk and folded rank
# normalisation), one sort and five FFT-based ESS for the full summary (DESIGN §8.2).

_TAIL_PROBS = (0.05, 0.95)
_MCSE_PROBS = np.array([0.1586553, 0.8413447])  # Φ(−1), Φ(1): a ±1 s.d. band of order statistics
_FFT_BLOCK = 1 << 22  # complex spectrum elements per FFT block (64 MiB): bounds host memory
_RANK_BLOCK = 1 << 21  # pooled draws per ranking block (bounds the int64 work arrays)


def _as_cdq(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """``(chains, draws, *S)`` → contiguous ``(Q, chains, draws)`` with ``Q = ∏S``, and ``S``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"expected an array (chains, draws, ...), got shape {x.shape}")
    trailing = x.shape[2:]
    q = int(np.prod(trailing, dtype=np.int64))
    flat = x.reshape(x.shape[0], x.shape[1], q)
    return np.ascontiguousarray(np.moveaxis(flat, 2, 0)), trailing


def _flat(a: np.ndarray) -> np.ndarray:
    """``(Q, m, n)`` → ``(Q, m·n)``: the pooled draws of every quantity (``Q = 0`` allowed)."""
    return a.reshape(a.shape[0], a.shape[1] * a.shape[2])


def _scalar(fn, chains: np.ndarray, *args) -> float:
    x = np.asarray(chains, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected chains of shape (chains, draws), got {x.shape}")
    return float(fn(x[:, :, None], *args)[0])


def _check_prob(prob: float) -> float:
    prob = float(prob)
    if not 0.0 < prob < 1.0:
        raise ValueError(f"quantile probability must lie in (0, 1), got {prob}")
    return prob


def _split_q(xq: np.ndarray) -> np.ndarray:
    """``(Q, m, n)`` → ``(Q, 2m, n // 2)``: every chain's first half, then every second half."""
    half = xq.shape[2] // 2
    return np.concatenate([xq[:, :, :half], xq[:, :, xq.shape[2] - half :]], axis=1)


def _normal_scores_q(z: np.ndarray) -> np.ndarray:
    """Rank normalisation per quantity, ``_rank_normalise`` row by row: Blom scores
    ``Φ⁻¹((r − 3/8)/(S + 1/4))`` of the pooled ranks ``r`` (ties averaged; NaN anywhere → NaN).

    One argsort per quantity. Average ranks are half-integers ``(first + last + 2)/2`` of a tie
    run's sorted positions, so the scores are read from a table of the ``2S − 1`` possible values.
    """
    q, m, n = z.shape
    size = m * n
    out = np.full((q, size), np.nan)
    flat = z.reshape(q, size)
    rows = np.flatnonzero(~np.isnan(flat).any(axis=1)) if size else np.empty(0, np.intp)
    if rows.size == 0:
        return out.reshape(z.shape)
    twice = np.arange(2, 2 * size + 1)  # 2 r
    table = stats.norm.ppf((twice / 2 - 0.375) / (size + 0.25))
    pos = np.arange(size)
    step = max(1, _RANK_BLOCK // size)
    for lo in range(0, rows.size, step):
        r = rows[lo : lo + step]
        block = flat[r]
        order = np.argsort(block, axis=1)
        srt = np.take_along_axis(block, order, axis=1)
        new = np.ones(srt.shape, bool)  # a tie run starts here
        np.not_equal(srt[:, 1:], srt[:, :-1], out=new[:, 1:])
        end = np.ones(srt.shape, bool)  # a tie run ends here
        end[:, :-1] = new[:, 1:]
        first = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        last = np.minimum.accumulate(np.where(end, pos, size - 1)[:, ::-1], axis=1)[:, ::-1]
        scores = np.empty(srt.shape)
        np.put_along_axis(scores, order, table[first + last], axis=1)
        out[r] = scores
    return out.reshape(z.shape)


def _rhat_q(z: np.ndarray) -> np.ndarray:
    """Potential scale reduction ``√(((n−1)/n·W + B/n)/W)`` per quantity; NaN when ``W ≤ 0``."""
    n = z.shape[2]
    within = z.var(axis=2, ddof=1).mean(axis=1)
    between = n * z.mean(axis=2).var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.sqrt(((n - 1) / n * within + between / n) / within)
    return np.where(within > 0, value, np.nan)


def _rhat_split(split: np.ndarray, z_bulk: np.ndarray) -> np.ndarray:
    """``max(bulk, folded)`` rank-normalised split R-hat from split chains and their scores."""
    bulk = _rhat_q(z_bulk)
    median = np.median(_flat(split), axis=1)
    with np.errstate(invalid="ignore"):  # ∞ − ∞ when more than half the draws are infinite
        folded_draws = np.abs(split - median[:, None, None])
    folded = _rhat_q(_normal_scores_q(folded_draws))
    return np.where(np.isnan(bulk) | np.isnan(folded), np.nan, np.maximum(bulk, folded))


def _geyer_ess(acov_mean: np.ndarray, mean_var: float, var_plus: float, total: int) -> float:
    """``S / τ̂``, ``τ̂ = −1 + 2 Σ_t ρ̂_t`` truncated by Geyer's initial positive and monotone
    sequences with the "improved estimate" term (Vehtari et al. 2021 eq. 10; the ``ess_bulk``
    recursion line for line), from chain-averaged autocovariances:
    ``ρ̂_t = 1 − (W − acov_t)/var⁺``."""
    if not var_plus > 0:
        return np.nan
    n = acov_mean.size

    def rho(t: int) -> float:
        return 1.0 - (mean_var - acov_mean[t]) / var_plus

    rho_hat = np.zeros(n)
    rho_even, rho_odd = 1.0, rho(1)
    rho_hat[0], rho_hat[1] = rho_even, rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:  # initial positive sequence
        rho_even, rho_odd = rho(t + 1), rho(t + 2)
        if rho_even + rho_odd >= 0:
            rho_hat[t + 1], rho_hat[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:  # improved estimate
        rho_hat[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:  # initial monotone sequence
        if rho_hat[t + 1] + rho_hat[t + 2] > rho_hat[t - 1] + rho_hat[t]:
            rho_hat[t + 1] = rho_hat[t + 2] = (rho_hat[t - 1] + rho_hat[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * rho_hat[: max_t + 1].sum() + rho_hat[max_t + 1 : max_t + 2].sum()
    return float(total / max(tau, 1.0 / np.log10(total)))


def _ess_q(z: np.ndarray) -> np.ndarray:
    """ESS of already split sequences ``z`` (``(Q, m, n)``, ``n ≥ 5``, no ±∞) per quantity.

    Autocovariances by FFT, ``acov_t = Σ_s z̃_s z̃_{s+t}/n`` per sequence; ``W`` the mean
    within-sequence variance and ``var⁺ = (n−1)/n·W + B/n`` (Vehtari et al. 2021 eqs. 3–4, 10).
    """
    q, m, n = z.shape
    centred = z - z.mean(axis=2, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    block = max(1, _FFT_BLOCK // (m * (size // 2 + 1)))
    acov = np.empty((q, m, n))
    for lo in range(0, q, block):
        spectrum = np.fft.rfft(centred[lo : lo + block], n=size, axis=2)
        acov[lo : lo + block] = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=2)[..., :n]
    acov /= n
    mean_var = np.ascontiguousarray(acov[:, :, 0]).mean(axis=1) * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + (z.mean(axis=2).var(axis=1, ddof=1) if m > 1 else 0.0)
    acov_mean = np.ascontiguousarray(acov.transpose(0, 2, 1)).mean(axis=2)  # acov[j, :, t].mean()
    return np.array(
        [_geyer_ess(acov_mean[j], mean_var[j], var_plus[j], m * n) for j in range(q)],
        dtype=np.float64,
    )


def _moving(flat: np.ndarray) -> np.ndarray:
    """Quantities without NaN draws whose draws are not all equal (``flat``: ``(Q, N ≥ 1)``)."""
    return ~np.isnan(flat).any(axis=1) & (flat.max(axis=1) > flat.min(axis=1))


def _ess_mean_q(xq: np.ndarray) -> np.ndarray:
    """ESS of the split chains on the original scale; NaN unless every draw is finite."""
    out = np.full(xq.shape[0], np.nan)
    if xq.shape[2] // 2 < 5:
        return out
    flat = _flat(xq)
    ok = _moving(flat) & np.isfinite(flat).all(axis=1)
    if ok.any():
        out[ok] = _ess_q(_split_q(xq[ok]))
    return out


def _quantiles_q(flat: np.ndarray, probs: tuple[float, ...]) -> np.ndarray:
    """Type-7 (linear) quantiles of the pooled draws, ``(len(probs), Q)``; NaN draws → NaN."""
    out = np.full((len(probs), flat.shape[0]), np.nan)
    ok = ~np.isnan(flat).any(axis=1) if flat.shape[1] else np.zeros(flat.shape[0], bool)
    if ok.any():
        with np.errstate(invalid="ignore"):  # ±∞ draws can make the interpolation NaN
            out[:, ok] = np.quantile(flat[ok], list(probs), axis=1)
    return out


def _indicator_ess_q(xq: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """ESS of ``1{x ≤ q̂}`` on split chains (``qv``: one threshold per quantity; Vehtari et al.
    2021 §4.3). A chain whose indicator never changes although its draws do (an atom at the
    quantile) gets the number of split draws, as in ArviZ; NaN thresholds, NaN draws and chains
    that never move give NaN."""
    out = np.full(xq.shape[0], np.nan)
    if xq.shape[2] // 2 < 5:
        return out
    ok = _moving(_flat(xq)) & ~np.isnan(qv)
    if not ok.any():
        return out
    split = _split_q((xq[ok] <= qv[ok, None, None]).astype(np.float64))
    flat = _flat(split)
    changes = flat.max(axis=1) > flat.min(axis=1)
    res = np.full(split.shape[0], float(flat.shape[1]))
    if changes.any():
        res[changes] = _ess_q(split[changes])
    out[ok] = res
    return out


def _mcse_quantile_sorted(srt: np.ndarray, ess: np.ndarray, prob: float) -> np.ndarray:
    """Half the distance between the order statistics at the ``Φ(∓1)`` quantiles of
    ``Beta(S·p + 1, S·(1 − p) + 1)`` (``srt``: pooled draws sorted per quantity, ``S = ess``)."""
    out = np.full(srt.shape[0], np.nan)
    ok = np.isfinite(ess)
    if ok.any():
        e = ess[ok]
        ppf = stats.beta.ppf(_MCSE_PROBS[:, None], e * prob + 1, e * (1 - prob) + 1)
        size = srt.shape[1]
        pos = ppf * size - 1
        lo = np.floor(np.maximum(pos[0], 0)).astype(np.intp)
        hi = np.ceil(np.minimum(pos[1], size - 1)).astype(np.intp)
        rows = srt[ok]
        idx = np.arange(e.size)
        with np.errstate(invalid="ignore"):  # ±∞ order statistics
            out[ok] = (rows[idx, hi] - rows[idx, lo]) / 2
    return out


def rhat_many(x: np.ndarray) -> np.ndarray:
    """:func:`rhat` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    xq, trailing = _as_cdq(x)
    out = np.full(xq.shape[0], np.nan)
    split = _split_q(xq)
    if split.shape[2] >= 2 and xq.shape[0]:
        out = _rhat_split(split, _normal_scores_q(split))
    return out.reshape(trailing)


def ess_bulk_many(x: np.ndarray) -> np.ndarray:
    """:func:`ess_bulk` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    xq, trailing = _as_cdq(x)
    out = np.full(xq.shape[0], np.nan)
    split = _split_q(xq)
    if split.shape[2] >= 5 and xq.shape[0]:
        out = _ess_q(_normal_scores_q(split))
    return out.reshape(trailing)


def rhat_ess_bulk_many(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rhat_many(x), ess_bulk_many(x))`` sharing the rank normalisation (the ``converged``
    flag's statistics, DESIGN §8.2)."""
    xq, trailing = _as_cdq(x)
    rhat_ = np.full(xq.shape[0], np.nan)
    ess = np.full(xq.shape[0], np.nan)
    split = _split_q(xq)
    if split.shape[2] >= 2 and xq.shape[0]:
        z = _normal_scores_q(split)
        rhat_ = _rhat_split(split, z)
        if split.shape[2] >= 5:
            ess = _ess_q(z)
    return rhat_.reshape(trailing), ess.reshape(trailing)


def ess_mean_many(x: np.ndarray) -> np.ndarray:
    """:func:`ess_mean` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    xq, trailing = _as_cdq(x)
    return _ess_mean_q(xq).reshape(trailing)


def ess_tail_many(x: np.ndarray) -> np.ndarray:
    """:func:`ess_tail` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    xq, trailing = _as_cdq(x)
    qv = _quantiles_q(_flat(xq), _TAIL_PROBS)
    low, high = (_indicator_ess_q(xq, qv[j]) for j in range(2))
    return np.minimum(low, high).reshape(trailing)


def mcse_mean_many(x: np.ndarray) -> np.ndarray:
    """:func:`mcse_mean` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    xq, trailing = _as_cdq(x)
    ess = _ess_mean_q(xq)
    out = np.full(xq.shape[0], np.nan)
    ok = np.isfinite(ess)
    if ok.any():
        out[ok] = _flat(xq[ok]).std(axis=1, ddof=1) / np.sqrt(ess[ok])
    return out.reshape(trailing)


def mcse_quantile_many(x: np.ndarray, q: float) -> np.ndarray:
    """:func:`mcse_quantile` of every quantity of ``x`` (``(chains, draws, *S)`` → ``S``)."""
    prob = _check_prob(q)
    xq, trailing = _as_cdq(x)
    flat = _flat(xq)
    ess = _indicator_ess_q(xq, _quantiles_q(flat, (prob,))[0])
    return _mcse_quantile_sorted(np.sort(flat, axis=1), ess, prob).reshape(trailing)


def ess_mean(chains: np.ndarray) -> float:
    """Effective sample size for the posterior mean: the ESS of the split chains on their
    original scale, without rank normalisation (Vehtari et al. 2021 §4.4; ArviZ ``"mean"``).
    Unlike the bulk ESS it is not invariant to monotone transforms; non-finite draws give NaN."""
    return _scalar(ess_mean_many, chains)


def ess_tail(chains: np.ndarray) -> float:
    """Tail effective sample size ``min(ESS(1{x ≤ q̂₀.₀₅}), ESS(1{x ≤ q̂₀.₉₅}))``, the quantile
    ESS of the 5% and 95% quantiles of the pooled draws (Vehtari et al. 2021 §4.3; ArviZ
    ``"tail"``)."""
    return _scalar(ess_tail_many, chains)


def mcse_mean(chains: np.ndarray) -> float:
    """Monte Carlo standard error of the posterior mean, ``sd/√ESS_mean`` with the ``sd`` of all
    draws (``ddof = 1``; Vehtari et al. 2021 §4.4)."""
    return _scalar(mcse_mean_many, chains)


def mcse_quantile(chains: np.ndarray, q: float) -> float:
    """Monte Carlo standard error of the ``q``-quantile (Vehtari et al. 2021 §4.3).

    With ``S`` the ESS of ``1{x ≤ q̂_q}`` and ``a₁, a₂`` the ``Φ(∓1)`` quantiles of
    ``Beta(S·q + 1, S·(1 − q) + 1)``, the MCSE is half the distance between the order statistics
    ``x_(⌊a₁N⌋)`` and ``x_(⌈a₂N⌉)`` (1-based, clipped to the ``N`` pooled draws).
    """
    return _scalar(mcse_quantile_many, chains, q)


def summary_many(
    x: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    *,
    rank_normalised: bool = True,
) -> dict[str, np.ndarray]:
    """Posterior summary and convergence diagnostics of every quantity of ``x``
    (``(chains, draws, *S)``), sharing the sorts and FFTs between the statistics.

    Keys (arrays of shape ``S``): ``mean``, ``sd`` (pooled, ``ddof = 1``), ``mcse_mean``,
    ``ess_mean``, ``q{100p}`` and ``mcse_q{100p}`` for each ``p`` in ``quantiles`` (e.g. ``q5``,
    ``mcse_q5``), ``rhat``, ``ess_bulk``, ``ess_tail``. Each equals the matching ``*_many``.
    ``rank_normalised=False`` leaves out ``rhat`` and ``ess_bulk`` (two argsorts and one FFT per
    quantity) for a caller that has them already, e.g. from :func:`rhat_ess_bulk_many`.
    """
    probs = tuple(_check_prob(p) for p in quantiles)
    xq, trailing = _as_cdq(x)
    nq = xq.shape[0]
    flat = _flat(xq)
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):  # ±∞ draws: ∞ − ∞ in the moments
        out["mean"] = flat.mean(axis=1) if flat.shape[1] else np.full(nq, np.nan)
        out["sd"] = flat.std(axis=1, ddof=1) if flat.shape[1] > 1 else np.full(nq, np.nan)
    out["ess_mean"] = _ess_mean_q(xq)
    ok = np.isfinite(out["ess_mean"])
    out["mcse_mean"] = np.full(nq, np.nan)
    out["mcse_mean"][ok] = out["sd"][ok] / np.sqrt(out["ess_mean"][ok])
    unique = tuple(dict.fromkeys(probs))
    every = tuple(dict.fromkeys(unique + _TAIL_PROBS))  # the requested quantiles come first
    qv = _quantiles_q(flat, every)
    ind = {p: _indicator_ess_q(xq, qv[j]) for j, p in enumerate(every)}
    srt = np.sort(flat, axis=1)
    for j, p in enumerate(unique):
        name = f"{100 * p:g}"
        out[f"q{name}"] = qv[j]
        out[f"mcse_q{name}"] = _mcse_quantile_sorted(srt, ind[p], p)
    if rank_normalised:
        split = _split_q(xq)
        z = _normal_scores_q(split) if split.shape[2] >= 2 and nq else None
        out["rhat"] = _rhat_split(split, z) if z is not None else np.full(nq, np.nan)
        out["ess_bulk"] = (
            _ess_q(z) if z is not None and split.shape[2] >= 5 else np.full(nq, np.nan)
        )
    out["ess_tail"] = np.minimum(ind[_TAIL_PROBS[0]], ind[_TAIL_PROBS[1]])
    return {k: v.reshape(trailing) for k, v in out.items()}
