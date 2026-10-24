"""The F1 state: chain 3214 of ``test_g_full[reduced_sv]`` at sweep 194, the committed fixture of
``tests/mcmc/blocks/test_contemp_lstsq.py::test_g_full_reduced_sv_failing_chain_is_fixed``.

Run offline, never in CI (``python -I`` keeps the repository off ``sys.path``)::

    uv run python -I tools/make_f1_state.py write   # tests/fixtures/f1_reduced_sv_chain3214_sweep194.npz
    uv run python -I tools/make_f1_state.py check   # replay again, compare with the committed file

The replay is the harness's (``tests/geweke/test_geweke_compositions.py``, "Known nightly
finding"): the 500-chain block 3000–3499 of stage 0 with the Geweke keys of
``validation.geweke.rep_keys``, vmapped over the block, each chain started from its prior draw and
run 194 successive-conditional sweeps with the previous Cholesky ``a``-row draw (forced through the
``contemp_conditional`` seam: the same operations as before the fix). The file holds the state
``θ⁽¹⁹⁴⁾`` and the data ``y⁽¹⁹⁴⁾`` that sweep 194 runs on, one array per leaf keyed
``theta.<leaf path>`` and ``data.<leaf path>``, and ``meta`` (JSON: the chain's coordinates, the
commit, machine and versions of the recording, and the status codes the recording produced).

The trajectory is a floating-point path and reproduces bitwise only where it was recorded (Apple
M5 Max, macOS arm64). It is not portable because the previous draw's failure is a rounding outcome:
along this chain equation 2's ``a``-row precision has ``cond(K_a) ≈ 1/ε``, and the Jacobi-scaled
Cholesky pivot of the formed Gram is 0 (status 1003) or a few ε (status 0) depending on the last bits
of the inputs. Elsewhere the replay can fail at another sweep or not at all, so ``write`` records
only a replay that reproduces the finding (status 0 before sweep 194, 1003 from the previous draw at
sweep 194, 0 from the production step) and ``check`` reports where a replay departs from the file.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import zlib
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.keys import Tier
from rai_bvar.engines.mcmc.state import leaf_path
from rai_bvar.validation.geweke import Part, part, rep_keys
from rai_bvar.validation.joint import joint_spec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "f1_reduced_sv_chain3214_sweep194.npz"
NAME, SEED, CHAIN, SWEEP, BLOCK0, BLOCK = "reduced_sv", 20261023, 3214, 194, 3000, 500


def replay() -> tuple[dict[str, np.ndarray], dict[str, int], int | None]:
    """The arrays of ``θ⁽¹⁹⁴⁾`` and ``y⁽¹⁹⁴⁾`` of the chain, the status codes of the replay and the
    first sweep before 194 at which the chain failed (``None``: none)."""
    spec = joint_spec(NAME, "full")
    old = joint_spec(
        NAME, "full", kernel_overrides={"contemp_conditional": compose(spec.static).structure
                                        .contemp_conditional}
    )  # fmt: skip
    keys_b = rep_keys(SEED, Tier.G, zlib.crc32(NAME.encode()) & 0xFFFFFFFF,
                      np.arange(BLOCK0, BLOCK0 + BLOCK))  # fmt: skip

    def run_to(key):
        th0 = old.prior_sample(part(key, Part.PRIOR))
        sim_key, ckey = part(key, Part.SIM), part(key, Part.STEP)

        def body(th, it):
            th = old.step(ckey, it, th, old.simulate(random.fold_in(sim_key, it), th))
            return th, th.status.code

        th, path = lax.scan(body, th0, jnp.arange(SWEEP, dtype=jnp.uint32))
        y = old.simulate(random.fold_in(sim_key, SWEEP), th)
        it = jnp.uint32(SWEEP)
        return th, y, old.step(ckey, it, th, y), spec.step(ckey, it, th, y), path

    pre, y, with_old, with_new, path = jax.jit(jax.vmap(run_to))(keys_b)
    j = CHAIN - BLOCK0
    arrays = {}
    for prefix, tree in (("theta", pre), ("data", y)):
        for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[f"{prefix}.{leaf_path(p)}"] = np.asarray(x[j])
    codes = {
        "before": int(pre.status.code[j]),
        "previous_draw": int(with_old.status.code[j]),
        "production": int(with_new.status.code[j]),
    }
    failed = np.flatnonzero(np.asarray(path[j]))
    return arrays, codes, (int(failed[0]) if failed.size else None)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _cpu() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True)
        return out.stdout.decode().strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def write() -> int:
    arrays, codes, _ = replay()
    if codes != {"before": 0, "previous_draw": 1003, "production": 0}:
        print(f"the replay does not reproduce the finding on this machine: {codes}")
        return 1
    meta = {
        "composition": NAME, "seed": SEED, "chain": CHAIN, "sweep": SWEEP,
        "block": [BLOCK0, BLOCK0 + BLOCK], "codes": codes,
        "commit": _git("rev-parse", "HEAD"), "src_modified": bool(_git("status", "--porcelain", "--", "src")),
        "machine": {"platform": platform.platform(), "cpu": _cpu()},
        "versions": {p: version(p) for p in ("jax", "jaxlib", "numpy")},
    }  # fmt: skip
    np.savez_compressed(FIXTURE, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {FIXTURE.relative_to(ROOT)}: {len(arrays)} arrays, {json.dumps(meta)}")
    return 0


def check() -> int:
    arrays, codes, first = replay()
    with np.load(FIXTURE) as f:
        meta = json.loads(str(f["meta"]))
        stored = {k: f[k] for k in f.files if k != "meta"}
    print(f"recorded on {meta['machine']} at {meta['commit'][:7]}: codes {meta['codes']}")
    print(f"this replay ({platform.platform()}, {_cpu()}): codes {codes}, first failure before "
          f"sweep {SWEEP}: {'none' if first is None else f'sweep {first}'}")  # fmt: skip
    same = codes == meta["codes"] and sorted(stored) == sorted(arrays)
    for k in sorted(stored):
        a, b = stored[k], arrays.get(k)
        if b is None or a.shape != b.shape or a.dtype != b.dtype:
            print(f"  {k}: layout differs")
            same = False
        elif not np.array_equal(a, b):
            n_bad = int(np.sum(~np.isfinite(b))) if b.dtype.kind == "f" else 0
            print(f"  {k}: {int(np.sum(a != b))} of {a.size} entries differ, {n_bad} non-finite")
            same = False
    print("bitwise identical" if same else "the replay departs from the committed state")
    return 0 if same else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write", help=f"replay chain {CHAIN} and write the fixture")
    sub.add_parser("check", help=f"replay chain {CHAIN} and compare with the committed fixture")
    args = parser.parse_args(argv)
    return write() if args.command == "write" else check()


if __name__ == "__main__":
    sys.exit(main())
