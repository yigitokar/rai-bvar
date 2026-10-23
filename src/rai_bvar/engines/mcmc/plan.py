"""The Gibbs order as data (DESIGN §4.1–§4.4, §4.8).

``build_plan(program)`` is the only place the order of a sweep is decided; ``sweep.py`` executes
the resulting :class:`Plan` step by step, and ``fit.plan.describe()`` prints it. Phase 2 plans:

structural form (``LinearStructural``, fusing; DESIGN §4.2)::

    equation   equation    EQ_MEAN       (δ_i, a_i) | λ_eff,i   writes eq.mean, eq.struct.a
    equation   residual    EQ_MEAN       e_i = y_i − Fθ_i        writes —
    equation   vol         EQ_VOL        VolBlock.sample_eq      writes eq.vol

reduced form (CCCM 2022; DESIGN §4.3)::

    pre        cccm        G_CCCM        Π | A, λ_eff            writes glob.mean.Pi
    equation   contemp     EQ_CONTEMP    a_j | Π, λ_eff,j        writes eq.struct.a
    equation   residual    EQ_MEAN       e_j = (UAᵀ)_j           writes —
    equation   vol         EQ_VOL        VolBlock.sample_eq      writes eq.vol

A non-fusing mean with its own state (Phase 4 BART) gets a ``mean`` step before ``contemp``; a
coefficient prior with local state (Phase 3) gets ``coef_local`` between ``residual`` and ``vol``.
Both are omitted when the corresponding state has no leaves (Phase 2: ``ZeroMean``, ``ReducedEq``,
``FixedDiagonal``). Inside ``vol`` the order is fixed by the block (DESIGN §4.4): transitory
``o → p → ν → τ``, then ``s → h → θ_SV`` centred, then non-centred.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax

from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.state import leaf_path, state_template

if TYPE_CHECKING:
    from rai_bvar.engines.mcmc.compose import Program

__all__ = ["RESIDUAL_INPUTS", "Plan", "Step", "build_plan", "check_invariants", "covers"]

Phase = Literal["pre", "equation", "post"]

RESIDUAL_INPUTS: frozenset[str] = frozenset({"glob.mean", "eq.mean", "eq.struct"})
"""State paths that enter the structural residuals (DESIGN §4.8 item 1)."""

_DESCRIPTIONS = {
    "equation": "(δ_i, a_i) | λ_eff,i joint Gaussian (fused contemporaneous draw)",
    "mean": "mean_i | a_i, λ_eff,i (non-fusing mean)",
    "contemp": "a_i | g_i, λ_eff,i: regress ỹ_i − g_i on −ỹ_{<i}",
    "residual": "e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i from the new draws",
    "coef_local": "local prior scales of the coefficients",
    "vol": "VolBlock.sample_eq: o → p → ν → τ, then s → h → θ_SV (atomic)",
}


@dataclass(frozen=True)
class Step:
    """One step of the sweep (DESIGN §4.1). ``reads``/``writes`` are dotted state paths; the
    write set is checked against the actual modifications by a tree diff (§4.8 item 4).
    ``block`` is the key-derivation id (the residual step draws nothing and reuses ``EQ_MEAN``)."""

    name: str
    block: Block
    phase: Phase
    reads: frozenset[str]
    writes: frozenset[str]


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]

    @property
    def pre(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.phase == "pre")

    @property
    def equation(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.phase == "equation")

    @property
    def post(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if s.phase == "post")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps)

    def describe(self) -> str:
        """The order of one sweep as a table (printed by ``fit.plan.describe()``)."""
        lines = [f"{'phase':<9} {'step':<11} {'block':<16} writes"]
        for s in self.steps:
            block = f"{s.block.name}({int(s.block)})"
            writes = ", ".join(sorted(s.writes)) or "—"
            lines.append(f"{s.phase:<9} {s.name:<11} {block:<16} {writes}")
            if s.name in _DESCRIPTIONS:
                lines.append(f"{'':<9} {'':<11} {'':<16} {_DESCRIPTIONS[s.name]}")
        if not self.post:
            lines.append("post      (none)")
        return "\n".join(lines)


def covers(path: str, target: str) -> bool:
    """Whether the dotted state paths ``path`` and ``target`` overlap (one is a prefix)."""
    return path == target or target.startswith(path + ".") or path.startswith(target + ".")


def _within(path: str, allowed: Sequence[str]) -> bool:
    """``path`` lies inside one of ``allowed`` (equal or a descendant)."""
    return any(path == a or path.startswith(a + ".") for a in allowed)


def _has_leaves(tree: Any) -> bool:
    return bool(jax.tree.leaves(tree))


def mean_has_state(program: Program) -> bool:
    """Whether the mean keeps per-equation state (``has_state`` on the kernel, else the template)."""
    flag = getattr(program.mean, "has_state", None)
    if flag is not None:
        return bool(flag)
    return _has_leaves(state_template(program.static).eq.mean)


def coef_has_local_state(program: Program) -> bool:
    """Whether the coefficient prior keeps local scales (``has_local_state``, else the template)."""
    flag = getattr(program.coef, "has_local_state", None)
    if flag is not None:
        return bool(flag)
    return _has_leaves(state_template(program.static).eq.coef)


def _global_name(g: Any) -> str:
    name = getattr(g, "name", None)
    if isinstance(name, str) and name:
        return name
    block = Block(int(g.block)).name
    return block[2:].lower() if block.startswith("G_") else block.lower()


def build_plan(program: Program) -> Plan:
    """The sweep order of ``program`` (DESIGN §4.2–§4.3); see the module docstring."""
    steps: list[Step] = []
    for g in program.global_pre:
        steps.append(
            Step(
                _global_name(g), Block(int(g.block)), "pre", frozenset(g.reads), frozenset(g.writes)
            )
        )
    common = frozenset({"eq.vol", "eq.mean", "eq.struct", "eq.coef", "glob"})
    if program.mean.fuses_contemp:
        steps.append(
            Step(
                "equation", Block.EQ_MEAN, "equation", common, frozenset({"eq.mean", "eq.struct.a"})
            )
        )
    else:
        if mean_has_state(program):
            steps.append(Step("mean", Block.EQ_MEAN, "equation", common, frozenset({"eq.mean"})))
        steps.append(
            Step("contemp", Block.EQ_CONTEMP, "equation", common, frozenset({"eq.struct.a"}))
        )
    steps.append(
        Step(
            "residual",
            Block.EQ_MEAN,
            "equation",
            frozenset({"eq.mean", "eq.struct", "glob.mean"}),
            frozenset(),
        )
    )
    if coef_has_local_state(program):
        steps.append(
            Step(
                "coef_local",
                Block.EQ_COEF_LOCAL,
                "equation",
                frozenset({"eq.coef", "eq.mean", "eq.struct", "glob.coef"}),
                frozenset({"eq.coef"}),
            )
        )
    steps.append(
        Step(
            "vol",
            Block.EQ_VOL,
            "equation",
            frozenset({"eq.vol", "glob.vol"}),
            frozenset({"eq.vol"}),
        )
    )
    for g in program.global_post:
        steps.append(
            Step(
                _global_name(g),
                Block(int(g.block)),
                "post",
                frozenset(g.reads),
                frozenset(g.writes),
            )
        )
    return Plan(tuple(steps))


def _static_violations(plan: Plan, program: Program) -> list[str]:
    out: list[str] = []
    phases = [s.phase for s in plan.steps]
    rank = {"pre": 0, "equation": 1, "post": 2}
    if any(rank[a] > rank[b] for a, b in itertools.pairwise(phases)):
        out.append("phases out of order: pre steps, then the equation body, then post steps")
    eq_steps = plan.equation
    vols = [k for k, s in enumerate(eq_steps) if s.name == "vol"]
    if len(vols) != 1:
        out.append(f"each equation needs exactly one vol step, found {len(vols)} (§4.8 item 2)")
    vol_pos = vols[0] if vols else len(eq_steps)
    for k, s in enumerate(eq_steps):  # item 1
        if k > vol_pos and any(covers(w, r) for w in s.writes for r in RESIDUAL_INPUTS):
            out.append(f"step {s.name!r} writes residual inputs after the vol step (§4.8 item 1)")
    for s in plan.post:
        if any(covers(w, r) for w in s.writes for r in RESIDUAL_INPUTS):
            out.append(
                f"post step {s.name!r} writes residual inputs {sorted(s.writes)} (§4.8 item 1)"
            )
    for s in plan.steps:  # item 3
        if s.name != "vol" and any(covers(w, "eq.vol") for w in s.writes):
            out.append(
                f"step {s.name!r} writes eq.vol, which only the vol step may write (§4.8 item 3)"
            )
        allowed = {
            "pre": ("glob",),
            "equation": ("eq",),
            "post": ("glob", "eq.coef"),
        }[s.phase]
        bad = sorted(w for w in s.writes if not _within(w, allowed))
        if bad:
            out.append(
                f"{s.phase} step {s.name!r} writes {bad}; allowed: {list(allowed)} (§4.8 item 3)"
            )
    names = {leaf_path(p).split(".")[-1] for p, _ in jax.tree_util.tree_flatten_with_path(
        state_template(program.static))[0]}  # fmt: skip
    if names & {"s", "s_idx", "indicators"}:
        out.append("mixture indicators must never be state (§4.8 item 2)")
    static = program.static
    scaled = static.struct.contemp_scaling == "lambda" or static.mean.coef_scaling == "lambda"
    if scaled and not (static.vol.coupled and static.vol.persistent == "constant"):
        out.append("λ-scaled prior blocks need a coupled Constant core (§4.8 item 5)")
    for g in program.global_steps:  # item 6
        if getattr(g, "marginal_of_indicators", False) is not True:
            out.append(
                f"global step {type(g).__name__} is not marginal of the indicators (§4.8 item 6)"
            )
    blocks = [int(s.block) for s in plan.pre + plan.post]
    if len(blocks) != len(set(blocks)):
        out.append(f"duplicate global step blocks {blocks}")
    return out


def check_invariants(plan: Plan, program: Program, *, probe: tuple[Any, ...] | None = None) -> None:
    """Raise ``ValueError`` listing every violated plan invariant (DESIGN §4.8).

    Static checks: items 1–3, 5 (static part) and 6. With ``probe = (ckey, it, st, data, prior,
    cache)`` (one chain) every step is also executed on its own at that state (``sweep.trace_steps``)
    and its actual modifications must lie inside its declared ``writes`` (item 4, the write-set tree
    diff); without a coupled core every step's ``Coupling`` must be exactly zero (item 5).
    """
    out = _static_violations(plan, program)
    if probe is not None:
        from rai_bvar.engines.mcmc.sweep import trace_steps

        for tr in trace_steps(program, plan, *probe):
            extra = sorted(
                p for p in tr.modified if not any(_within(p, [w]) for w in tr.step.writes)
            )
            if extra:
                out.append(
                    f"step {tr.step.name!r} modified {extra} outside its declared writes "
                    f"{sorted(tr.step.writes)} (§4.8 item 4)"
                )
            if tr.coupling_nonzero and not program.static.vol.coupled:
                out.append(f"step {tr.step.name!r} returned a non-zero Coupling (§4.8 item 5)")
    if out:
        raise ValueError("plan invariants violated:\n  " + "\n  ".join(out))
