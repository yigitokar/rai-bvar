"""Reference datasets: FRED-MD and FRED-QD, downloaded on demand with pinned vintages (SPEC §5.8)."""

from rai_bvar.datasets.fred import REGISTRY, VARIABLE_SETS, load, parse_fred_csv

__all__ = ["REGISTRY", "VARIABLE_SETS", "load", "parse_fred_csv"]
