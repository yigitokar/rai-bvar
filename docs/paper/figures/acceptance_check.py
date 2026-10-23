"""Is a Clark--Ravazzolo report the registered acceptance run? (no dependencies)

Shared by the acceptance figures (through ``make_common``) and by ``docs/paper/fill_acceptance.py``,
so that every consumer of ``benchmarks/reports/clark_ravazzolo_acceptance.json`` applies the same
test before it prints a number from it.
"""

from __future__ import annotations

REGISTERED_COMMIT = "c823d20"  # the commit that adds the pre-registration and the ACCEPTANCE config


def acceptance_problems(rep: dict) -> list[str]:
    """Why ``rep`` is not the registered acceptance report (empty when it is): the header fields
    of OUTLINE TODO 2 (name, estimator, forecast draws, 4 x 20,000 with burn 1000, seed 0, commit
    c823d20, clean tree, no ``pending``) plus the remaining checks of ``acceptance_gaps()`` in
    ``benchmarks/clark_ravazzolo.py`` at c823d20 (origin, ordering and noise steps of the
    registered config; all 24 orderings over every origin; the structural pair). Unlike
    ``acceptance_gaps()``, which accepts any known commit, this check also pins the commit, and it
    compares only chains, draws and burn of ``config.mcmc`` (not the whole settings dict)."""
    cfg, ver = rep.get("config", {}), rep.get("versions", {})
    mcmc = cfg.get("mcmc", {})
    out = []
    if cfg.get("name") != "acceptance":
        out.append(f"config.name = {cfg.get('name')!r}")
    if cfg.get("lpd_method") != "integrated":
        out.append(f"config.lpd_method = {cfg.get('lpd_method')!r}")
    if cfg.get("draws") != 8000:
        out.append(f"config.draws = {cfg.get('draws')!r}")
    if (mcmc.get("chains"), mcmc.get("draws"), mcmc.get("burn")) != (4, 20000, 1000):
        out.append(f"config.mcmc = {mcmc.get('chains')} x {mcmc.get('draws')} (burn {mcmc.get('burn')})")
    for key in ("origin_step", "ordering_step", "noise_reps"):  # CONFIGS["acceptance"] at c823d20
        if cfg.get(key) != 1:
            out.append(f"config.{key} = {cfg.get(key)!r}")
    if cfg.get("seed") != 0:
        out.append(f"config.seed = {cfg.get('seed')!r}")
    if not str(ver.get("git_commit", "")).startswith(REGISTERED_COMMIT):
        out.append(f"versions.git_commit = {ver.get('git_commit')!r}")
    if ver.get("git_dirty") is not False:
        out.append(f"versions.git_dirty = {ver.get('git_dirty')!r}")
    if "pending" in rep:
        out.append(f"pending = {rep['pending']!r}")
    ords = rep.get("orderings") or {}
    n_full = (rep.get("n_origins") or {}).get("full")
    if len(ords.get("per_ordering", [])) != 24 or len(ords.get("origins", [])) != n_full:
        out.append("the 24-ordering stage is incomplete")
    if not rep.get("structural"):
        out.append("the structural pair is missing")
    return out
