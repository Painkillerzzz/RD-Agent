"""Thin wrappers around the zoo repo: zoo_val.check and blackbox/local/local_eval.py, run with the zoo python."""
from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
from pathlib import Path

from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S


REMOTE = bool(S.ssh.strip())


def _p(x: str) -> Path:
    if REMOTE:  # remote paths: expand "~" on the remote side, keep them as given here
        return Path(x)
    return Path(os.path.expanduser(x)).resolve()


def _rp(x) -> str:
    """Path for a remote shell command, with ~ left for the remote shell to expand."""
    s = str(x)
    return s if not s.startswith("~") else "$HOME" + s[1:]


def ssh(cmd: str, stdin: str | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", *shlex.split(S.ssh), cmd], input=stdin, capture_output=True,
                          text=True, timeout=timeout)


def read_text(path: Path) -> str:
    if not REMOTE:
        return path.read_text()
    p = ssh(f'cat "{_rp(path)}"')
    if p.returncode != 0:
        raise FileNotFoundError(f"{path}: {p.stderr[-500:]}")
    return p.stdout


def push_spec(local: Path) -> str:
    """Copy a spec to the evaluation machine; returns the path to use there."""
    if not REMOTE:
        return str(local)
    remote = f"{_rp(WORK)}/specs/{local.name}"
    p = ssh(f'mkdir -p "{_rp(WORK)}/specs" && cat > "{remote}"', stdin=local.read_text())
    if p.returncode != 0:
        raise RuntimeError(f"ssh copy failed: {p.stderr[-500:]}")
    return remote


def run(argv: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    envs = {"PYTHONPATH": f"{_rp(REPO)}/validation", "ZOO_DATA": _rp(DATA)}
    if not REMOTE:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env={**os.environ, **envs},
                           cwd=str(cwd))
    else:
        line = " ".join(f'{k}="{v}"' for k, v in envs.items()) + " " + " ".join(
            a if a.startswith("$HOME") else shlex.quote(a) for a in argv)
        p = ssh(f'cd "{_rp(cwd)}" && {line}', timeout=timeout)
    return p.returncode, p.stdout + p.stderr


# keep the venv python unresolved: resolving its symlink would drop the venv
REPO, DATA, WORK, PY = _p(S.repo), _p(S.data), _p(S.work), os.path.expanduser(S.python)


def parent_row(stem: str) -> dict:
    import io
    for r in csv.DictReader(io.StringIO(read_text(REPO / "research_results" / "specs.csv"))):
        if r["stem"] == stem:
            return r
    return {}


def parent_code(stem: str) -> str:
    return read_text(REPO / "research_results" / "specs" / f"{stem}.py")


def family_of(stem: str) -> str:
    return S.family or parent_row(stem).get("family") or stem


def check(spec: Path, timeout: int = 900) -> tuple[bool, str]:
    """zoo_val.check on the synthetic fixture (schema, shape, truncate() causal check)."""
    rc, out = run([_rp(PY), "-m", "zoo_val.check", push_spec(spec)], REPO / "validation", timeout)
    ok = rc == 0
    try:
        ok = ok and json.loads(out[out.index("{"):out.rindex("}") + 1]).get("ok", False)
    except Exception:  # noqa: BLE001
        pass
    return ok, out[-6000:]


def evaluate(spec: Path, family: str, note: str = "", parent: str = "") -> dict:
    """Full devval run (fixture check, real-panel causal check, DEV, VAL, robustness). Returns the agent view."""
    out = WORK / "agent_views" / f"{spec.stem}.json"
    if REMOTE:
        ssh(f'mkdir -p "{_rp(WORK)}/agent_views"')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    remote_spec = push_spec(spec)
    cmd = [_rp(PY), f"{_rp(REPO)}/blackbox/local/local_eval.py", "--data", _rp(DATA), "--work", f"{_rp(WORK)}/sealed/eval",
           "--family", family, "--proposer", "rdagent", "--parent", parent, "--note", note[:480],
           "--agent-json", _rp(out), "--dev-only", "--sealed-json", _rp(WORK / "sealed" / f"{spec.stem}.json"),
           remote_spec]
    rc, log = run(cmd, REPO, S.eval_timeout)
    try:
        return json.loads(read_text(out))
    except Exception:  # noqa: BLE001
        return {"status": "failed", "error": log[-3000:]}


def score(view: dict | None) -> float | None:
    """DEV-only objective (bandit reward / ranking hint): mean of DEV Sharpe and the worst DEV Sharpe under +-20%
    parameter moves; None unless the run is ok and DEV t >= 2. Acceptance itself is opt_gate.decide (gate.py)."""
    if not view or view.get("status") != "ok":
        return None
    dev = view.get("dev") or {}
    if (dev.get("t_stat") or -9) < 2:
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
    f = lambda x, p=3: "n/a" if x is None else f"{x:.{p}f}"  # noqa: E731
    ric = (d.get("rank_ic") or {})
    return (f"DEV sharpe={f(d.get('sharpe'))} t={f(d.get('t_stat'), 2)} maxdd={f(d.get('max_drawdown'))} "
            f"gross_sharpe={f(d.get('gross_sharpe'))} turnover_ann={f(d.get('turnover_ann'), 1)} "
            f"skew={f(d.get('skew'), 2)} kurt={f(d.get('kurt'), 1)} rank_ic={f(ric.get('mean'), 4)} "
            f"by_year={d.get('sharpe_by_year')} | DEV perturb_min={f(view.get('dev_perturb_min'))} "
            f"fold_pos={view.get('dev_fold_pos')} | dev_objective={f(score(view))}")
