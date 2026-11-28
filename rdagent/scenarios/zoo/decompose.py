"""One-off LLM split of a ts-engine spec with cross-sectional logic into factor / model / portfolio layers.

The split is accepted only if the assembled spec passes zoo_val.check AND reproduces the parent's target positions
exactly (tools/repro_check.py, synthetic data only: no DEV/VAL numbers are involved). It is cached per parent in
<work>/fm_seeds/<parent>.json so a resumed or restarted loop does not pay for it again.
"""
from __future__ import annotations

import json
from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo import fm, llm

TOOLS = Path(__file__).parent / "tools"
MAX_TRIES = 3

SYSTEM = """You are an expert Python quant developer. You refactor a trading strategy spec WITHOUT changing its
behaviour, into three layers, so that a research loop can later change one layer at a time:
1. factors: one function per raw cross-sectional score the strategy ranks on,
   `def fac_<name>(bundle, params) -> pd.DataFrame` (ts x symbol, same index/columns as close). A factor is the
   score itself (e.g. a flow ratio, a positioning divergence), NOT a selection mask or position.
2. model: `def combine(F, bundle, params) -> dict` mapping each portfolio sleeve name to the score that sleeve ranks
   on. For the refactor it just routes, e.g. {"flow": F["flow"], "positioning": F["positioning"]}.
3. portfolio: `def portfolio(S, F, bundle, params) -> pd.DataFrame` that turns the sleeve scores into the exact
   target positions of the original signal(): universe/liquidity filter, names per side, clocks and cohorts, sleeve
   budgets, risk filters and scaling. It may use F for non-ranked inputs (filters) if needed.
The original module is available unchanged with SPEC renamed _PARENT_SPEC and signal renamed _parent_signal, so you
may call its private helpers (e.g. _tails, _aux, _clock_mask). Do not call _parent_signal. Keep every param name.
The output of portfolio() must be IDENTICAL to the original signal() for any data and any params.
Return ONE ```python block with all functions (helpers prefixed with an underscore allowed) and nothing else."""


def _cache_path(parent: str) -> Path:
    return ev.WORK / "fm_seeds" / f"{parent}.json"


def _write(path: Path, text: str) -> None:
    if ev.REMOTE:
        p = ev.ssh(f'mkdir -p "{ev._rp(path.parent)}" && cat > "{ev._rp(path)}"', stdin=text)
        if p.returncode != 0:
            raise RuntimeError(f"ssh write failed: {p.stderr[-500:]}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def repro(parent_spec: Path, cand_spec: Path, timeout: int = 900) -> dict:
    script = TOOLS / "repro_check.py"
    remote_script = ev.push_spec(script) if ev.REMOTE else str(script)
    rc, out = ev.run([ev._rp(ev.PY), remote_script, ev.push_spec(parent_spec), ev.push_spec(cand_spec)],
                     ev.REPO / "validation", timeout)
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1])
    except Exception:  # noqa: BLE001
        return {"equal": False, "error": out[-3000:]}


def _state_from(code: str, base: str, spec: dict, params: dict) -> dict:
    fns = fm.split_functions(code)
    for need in ("combine", "portfolio"):
        if need not in fns:
            raise ValueError(f"missing def {need}(...)")
    facs = {n[4:]: src for n, src in fns.items() if n.startswith("fac_")}
    if not facs:
        raise ValueError("no fac_<name> functions")
    helpers = "".join(src + "\n" for n, src in fns.items()
                      if n not in ("combine", "portfolio") and not n.startswith("fac_"))
    # private helpers of the refactor are kept as a fixed layer (factors and portfolio may both use them)
    return {"base": base, "helpers": helpers, "factors": facs, "model": fns["combine"], "portfolio": fns["portfolio"],
            "spec": spec, "params": params}


def decompose_ts(parent: str, parent_code: str, workdir: Path) -> dict:
    """State for a ts parent: cached, else LLM split + exact reproduction check (raises if it never reproduces)."""
    try:
        st = json.loads(ev.read_text(_cache_path(parent)))
        if st.get("portfolio") and st.get("factors"):
            logger.info(f"zoo_fm: using cached ts split of {parent} (factors {list(st['factors'])})")
            return st
    except Exception:  # noqa: BLE001
        pass
    base, spec, params = fm.base_of(parent_code)
    workdir.mkdir(parents=True, exist_ok=True)
    parent_path = workdir / f"{parent}.py"
    parent_path.write_text(parent_code)
    prompt = f"# Original spec\n```python\n{parent_code}\n```\nRefactor it into the three layers."
    msgs: list = []
    for i in range(MAX_TRIES):
        resp = llm.chat(prompt, SYSTEM, former_messages=msgs)
        msgs += [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
        from rdagent.scenarios.zoo.developer import extract_code
        try:
            st = _state_from(extract_code(resp), base, spec, params)
        except Exception as e:  # noqa: BLE001
            prompt = f"Could not use your answer: {e}. Return the complete block again."
            continue
        cand = workdir / f"split{i}-{parent}.py"[:200]
        cand.write_text(fm.assemble(st, spec.get("id", parent), note="exact ts split of the parent."))
        ok, log = ev.check(cand)
        if not ok:
            prompt = f"The assembled spec failed zoo_val.check. Fix it.\n```\n{log[-3000:]}\n```"
            continue
        r = repro(parent_path, cand)
        logger.info(f"zoo_fm ts split try {i + 1}: equal={r.get('equal')} max_abs_diff={r.get('max_abs_diff')} "
                    f"active_frac={r.get('active_frac')}")
        if r.get("equal") and (r.get("active_frac") or 0) > 0.01:
            _write(_cache_path(parent), json.dumps(st))
            return st
        prompt = ("Your refactor does not reproduce the original positions on synthetic data "
                  f"(NaN and 0 count as flat). Report:\n```\n{json.dumps(r)[:3000]}\n```\nFind the difference "
                  "and return the complete corrected block.")
    raise ValueError(f"ts split of {parent} did not reproduce the parent after {MAX_TRIES} tries")
