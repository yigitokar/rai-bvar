"""Model components along the ablation axes (SPEC §4.4; DESIGN §1.2).

Only the user-facing specifications are imported here (no JAX, DESIGN §1.1 rule 5); the MCMC
kernels in the subpackages ``structure``, ``mean``, ``coef`` and ``vol`` are loaded lazily by
``engines/mcmc/registry.py``.
"""

from rai_bvar.components.specs import SV, Constant, Linear

__all__ = ["SV", "Constant", "Linear"]
