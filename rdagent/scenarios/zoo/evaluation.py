"""Thin wrappers around the zoo repo: zoo_val.check and blackbox/local/local_eval.py, run with the zoo python."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S


def _p(x: str) -> Path:
    return Path(os.path.expanduser(x)).resolve()


# keep the venv python unresolved: resolving its symlink would drop the venv
REPO, DATA, WORK, PY = _p(S.repo), _p(S.data), _p(S.work), os.path.expanduser(S.python)


def parent_row(stem: str) -> dict:
    with open(REPO / "research_results" / "specs.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["stem"] == stem:
                return r
    return {}


def parent_code(stem: str) -> str:
    return (REPO / "research_results" / "specs" / f"{stem}.py").read_text()


def family_of(stem: str) -> str:
    return S.family or parent_row(stem).get("family") or stem


def env() -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO / "validation"), "ZOO_DATA": str(DATA)}


def check(spec: Path, timeout: int = 900) -> tuple[bool, str]:
    """zoo_val.check on the synthetic fixture (schema, shape, truncate() causal check)."""
    p = subprocess.run([PY, "-m", "zoo_val.check", str(spec)], capture_output=True, text=True, timeout=timeout,
                       env=env(), cwd=str(REPO / "validation"))
    out = (p.stdout + p.stderr)[-6000:]
    ok = p.returncode == 0
    try:
        ok = ok and json.loads(p.stdout[p.stdout.index("{"):]).get("ok", False)
    except Exception:  # noqa: BLE001
        pass
    return ok, out


def evaluate(spec: Path, family: str, note: str = "", parent: str = "") -> dict:
    """Full devval run (fixture check, real-panel causal check, DEV, VAL, robustness). Returns the agent view."""
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "agent_views" / f"{spec.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(REPO / "blackbox" / "local" / "local_eval.py"), "--data", str(DATA), "--work", str(WORK / "eval"),
           "--family", family, "--proposer", "rdagent", "--parent", parent, "--note", note[:480],
           "--agent-json", str(out), str(spec)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=S.eval_timeout, env=env(), cwd=str(REPO))
    if not out.exists():
        return {"status": "failed", "error": (p.stdout + p.stderr)[-3000:]}
    return json.loads(out.read_text())


def score(view: dict | None) -> float | None:
    """DEV-only objective: mean of DEV Sharpe and the worst DEV Sharpe under +-20% parameter moves.
    Disqualified (None) unless the run is ok, DEV t >= 2 and VAL is not 'fail'."""
    if not view or view.get("status") != "ok":
        return None
    dev = view.get("dev") or {}
    if (dev.get("t_stat") or -9) < 2 or ((view.get("val") or {}).get("bucket") == "fail"):
        return None
    sh, pm = dev.get("sharpe"), view.get("dev_perturb_min")
    if sh is None:
        return None
    return 0.5 * sh + 0.5 * (pm if pm is not None else sh)


def brief(view: dict | None) -> str:
    if not view:
        return "no result"
    if view.get("status") != "ok":
        err = view.get("error") or view.get("check") or view.get("real_causal") or ""
        return f"status={view.get('status')} {str(err)[:800]}"
    d = view.get("dev") or {}
    g = view.get("gates") or {}
    return (f"DEV sharpe={d.get('sharpe'):.3f} t={d.get('t_stat'):.2f} maxdd={d.get('max_drawdown'):.3f} "
            f"turnover_ann={d.get('turnover_ann'):.1f} gross={d.get('gross_exposure_avg')} "
            f"by_year={d.get('sharpe_by_year')} | DEV perturb_min={view.get('dev_perturb_min')} "
            f"fold_pos={view.get('dev_fold_pos')} | VAL bucket={(view.get('val') or {}).get('bucket')} | "
            f"verdict={view.get('verdict')} gates={ {k: v for k, v in g.items()} } | score={score(view)}")
