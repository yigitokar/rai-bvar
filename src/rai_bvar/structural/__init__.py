"""Structural analysis: recursive impulse responses and variance decompositions (SPEC §6.3)."""

from rai_bvar.structural.irf import IRFResult, fevd, ma_coefficients, recursive_irf

__all__ = ["IRFResult", "fevd", "ma_coefficients", "recursive_irf"]
