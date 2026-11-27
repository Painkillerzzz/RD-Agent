"""Acceptance through the zoo optimisation gate (research_opt/opt_gate.py in the zoo-opt checkout).

The LLM never sees VAL: the evaluator writes a DEV-only agent view plus a sealed record (VAL numbers, gates, BTC
beta) that only this module reads. A candidate first has to clear the DEV margin; only then is one VAL query charged
(opt_gate.charge_val_query, at most MAX_VAL_QUERIES per strategy and round) and opt_gate.decide() called. What goes
back to the LLM is decide()'s status, coarse reasons and flags.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S

_G = None


class ValBudgetExhausted(RuntimeError):
    pass


def G():
    global _G
    if _G is None:
        d = Path(S.gate_dir).expanduser()
        for p in (str(d), str(ev.REPO / "validation")):
            if p not in sys.path:
                sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location("opt_gate", d / "opt_gate.py")
        _G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_G)
    return _G


def sealed(stem: str) -> dict | None:
    p = ev.WORK / "sealed" / f"{stem}.json"
    try:
        d = json.loads(ev.read_text(p))
    except Exception:  # noqa: BLE001
        return None
    try:
        G().audit_sealed_read(p, "rdagent.zoo.gate")
    except Exception:  # noqa: BLE001
        pass
    return d


def screen_gates(rec: dict) -> dict | None:
    """Only the screening gates opt_gate compares (VAL-threshold gates are its own job)."""
    g = rec.get("gates") or {}
    if not g:
        return None
    return {k: g[k] for k in G().SCREEN_GATES if isinstance(g.get(k), bool)}


def strategy_key() -> str:
    return f"rdagent:{S.parent}"


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _part(d: dict, **keys) -> dict:
    return {k: v for k, v in keys.items() if _num(v) is not None}


MAX_LEVERAGE_RATIO = 2.0


def same_vol(cand: dict, parent: dict) -> tuple[dict, list[str], list[str]]:
    """Same-volatility comparison (agreed 2026-11-27): the candidate's net series is scaled by
    L = parent DEV vol / candidate DEV vol (VAL keeps the DEV factor); drawdown (arithmetic cumulative, exact under
    scaling), BTC beta and annual return are compared after scaling. Returns (scaled candidate record, dev reasons,
    val reasons)."""
    c = {"dev": dict(cand["dev"]), "val": dict(cand["val"]), "gates": cand.get("gates")}
    pv, cv = _num(parent["dev"].get("ann_vol")), _num(cand["dev"].get("ann_vol"))
    if not pv or not cv:
        return c, [], []
    L = pv / cv
    c["leverage"] = L
    dev_r, val_r = [], []
    if L > MAX_LEVERAGE_RATIO:
        dev_r.append("dev_leverage")
    for part in ("dev", "val"):
        d = c[part]
        for k in ("dd_arith", "beta_btc", "ann_return"):
            if _num(d.get(k)) is not None:
                d[k] = d[k] * L
        cr, pr = _num(d.get("ann_return")), _num(parent[part].get("ann_return"))
        if cr is not None and pr is not None and cr < pr:
            (dev_r if part == "dev" else val_r).append(f"{part}_return")
    return c, dev_r, val_r


def decide(name: str, cand: dict, parent: dict, original: dict, round_id: str) -> dict:
    """cand/parent/original are sealed records. Returns {status, pass, reasons, flags} (no numbers)."""
    if _num(cand["dev"].get("sharpe")) is None or _num(parent["dev"].get("sharpe")) is None:
        return {"name": name, "status": "fail", "pass": False, "reasons": ["dev_missing"], "flags": []}
    cand, dev_extra, val_extra = same_vol(cand, parent)
    cd, pd_, cv, pv = cand["dev"], parent["dev"], cand["val"], parent["val"]
    if cd["sharpe"] < pd_["sharpe"] + G().DEV_MARGIN or dev_extra:  # DEV screen first: no VAL query is spent
        rs = (["dev_margin"] if cd["sharpe"] < pd_["sharpe"] + G().DEV_MARGIN else []) + dev_extra
        return {"name": name, "status": "fail", "pass": False, "reasons": rs, "flags": ["val_not_queried"]}
    try:
        G().charge_val_query(strategy_key(), round_id)
    except G().ValBudgetExceeded as e:
        raise ValBudgetExhausted(str(e)) from e
    dd = lambda d: d.get("dd_arith") if _num(d.get("dd_arith")) is not None else d.get("max_drawdown")  # noqa: E731
    dev = _part(cd, cand=cd["sharpe"], parent=pd_["sharpe"], cand_beta=cd.get("beta_btc"), parent_beta=pd_.get("beta_btc"))
    val = _part(cv, cand=cv.get("sharpe"), parent=pv.get("sharpe"), original=(original.get("val") or {}).get("sharpe"),
                cand_dd=dd(cv), parent_dd=dd(pv), cand_beta=cv.get("beta_btc"), parent_beta=pv.get("beta_btc"))
    if "cand" not in val or "parent" not in val:
        return {"name": name, "status": "fail", "pass": False, "reasons": ["val_check_failed"], "flags": ["val_missing"]}
    kw = {"strategy": strategy_key()}
    if _num(cv.get("n_obs")):
        kw["val_n_obs"] = int(cv["n_obs"])
    try:
        res = G().decide(name, dev, val, parent_gates=screen_gates(parent), cand_gates=screen_gates(cand), **kw)
    except TypeError:  # older opt_gate without val_n_obs
        kw.pop("val_n_obs", None)
        res = G().decide(name, dev, val, parent_gates=screen_gates(parent), cand_gates=screen_gates(cand), **kw)
    out = {k: res.get(k) for k in ("name", "status", "pass", "reasons", "flags")}
    # opt_gate (8500adc6) folds val_* into val_check_failed but lets "below_original" through: fold it too
    rs = [r for r in (out["reasons"] or []) if r != "below_original"]
    if len(rs) != len(out["reasons"] or []) and "val_check_failed" not in rs:
        rs.append("val_check_failed")
    if val_extra and "val_check_failed" not in rs:  # same-vol VAL return floor failed: folded like the other VAL checks
        rs.append("val_check_failed")
    out["reasons"] = rs
    if val_extra or rs:
        out["status"], out["pass"] = "fail", False
    logger.info(f"opt_gate {name}: {out['status']} {out['reasons']}")  # no numbers in logs
    return out


def recheck(stem: str, parent_stem: str, original_stem: str | None = None) -> dict:
    """Re-run the acceptance rule for an already accepted candidate under the current rules (no VAL query is
    charged: it re-reads numbers that were queried when the candidate was accepted). Returns pass/fail + reasons."""
    cand, par = sealed(stem), sealed(parent_stem)
    orig = sealed(original_stem or parent_stem)
    if not (cand and par and orig):
        return {"name": stem, "status": "fail", "pass": False, "reasons": ["missing_sealed_record"], "flags": []}
    real = G().charge_val_query
    G().charge_val_query = lambda *a, **k: 0
    try:
        return decide(f"{stem}-recheck", cand, par, orig, "recheck")
    finally:
        G().charge_val_query = real


if __name__ == "__main__":  # python -m rdagent.scenarios.zoo.gate STEM PARENT [ORIGINAL]
    print(json.dumps(recheck(*sys.argv[1:4])))
