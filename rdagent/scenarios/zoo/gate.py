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


def decide(name: str, cand: dict, parent: dict, original: dict, round_id: str) -> dict:
    """cand/parent/original are sealed records. Returns {status, pass, reasons, flags} (no numbers)."""
    cd, pd_, cv, pv = cand["dev"], parent["dev"], cand["val"], parent["val"]
    if _num(cd.get("sharpe")) is None or _num(pd_.get("sharpe")) is None:
        return {"name": name, "status": "fail", "pass": False, "reasons": ["dev_missing"], "flags": []}
    if cd["sharpe"] < pd_["sharpe"] + G().DEV_MARGIN:  # DEV screen first: no VAL query is spent
        return {"name": name, "status": "fail", "pass": False, "reasons": ["dev_margin"], "flags": ["val_not_queried"]}
    try:
        G().charge_val_query(strategy_key(), round_id)
    except G().ValBudgetExceeded as e:
        raise ValBudgetExhausted(str(e)) from e
    dev = _part(cd, cand=cd["sharpe"], parent=pd_["sharpe"], cand_beta=cd.get("beta_btc"), parent_beta=pd_.get("beta_btc"))
    val = _part(cv, cand=cv.get("sharpe"), parent=pv.get("sharpe"), original=(original.get("val") or {}).get("sharpe"),
                cand_dd=cv.get("max_drawdown"), parent_dd=pv.get("max_drawdown"),
                cand_beta=cv.get("beta_btc"), parent_beta=pv.get("beta_btc"))
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
    out["reasons"] = rs
    logger.info(f"opt_gate {name}: {out['status']} {out['reasons']}")  # no numbers in logs
    return out
