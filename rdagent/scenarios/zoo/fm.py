"""Factor + model + portfolio decomposition of a zoo spec, and the arm-selection bandit.

A state is {"base": parent module (renamed), "factors": {name: code of fac_<name>}, "model": code of combine(),
"spec": SPEC overrides, "params": params}. assemble(state) renders ONE ordinary zoo spec module, so the evaluator,
zoo_val.check and the score are exactly the same as for the plain spec loop.

xs parents (engine=xs): the parent signal is factor "parent", the model is a rank mean, the xs engine builds the
portfolio from the score.
ts parents with cross-sectional logic (engine=ts, positions built by ranking assets): the state also has
"portfolio": code of portfolio(S, F, bundle, params) -> target positions. combine() returns a dict of sleeve scores
{sleeve: ts x symbol}; portfolio() turns them into positions the way the parent did (universe, names per side,
cohorts, sleeve budgets). The split is written once by the LLM (decompose.py) and accepted only if it reproduces
the parent's positions exactly on synthetic data.
"""
from __future__ import annotations

import ast
import json
import math
import pprint
import random
import re

ARMS = ("factor", "model", "portfolio")
MAX_FACTORS = 8

MODEL_RANK_MEAN = '''def combine(F, bundle, params):
    """Equal-weight mean of cross-sectional percentile ranks (sign carried by each factor)."""
    ranks = [f.rank(axis=1, pct=True) - 0.5 for f in F.values()]
    out = ranks[0].copy()
    for r in ranks[1:]:
        out = out.add(r, fill_value=0.0)
    return out / len(ranks)
'''

MODEL_GUIDE = """combine(F, bundle, params) gets F = {name: ts x symbol DataFrame} (one per factor, same index/columns as
close) and returns ONE ts x symbol score (higher = long). Options: weighted rank mean (weights as numeric params),
sign/IC-weighted mean where the weights are estimated ONLY on past rows (expanding/rolling, shifted by the label
horizon), or a purged walk-forward model (Ridge / HistGradientBoosting / ExtraTrees from sklearn) refit at most
monthly or quarterly on formation rows whose forward-return label ended strictly before the fold start, trained on
cross-sectional ranks, deterministic (fixed random_state). Labels may use shift(-k) ONLY inside that purged training
construction. Keep it fast: the full evaluation reruns the spec ~15 times."""


TS_MODEL_GUIDE = """combine(F, bundle, params) gets F = {name: ts x symbol DataFrame} (one per factor) and returns a
dict {sleeve: ts x symbol score} with exactly the sleeve keys that portfolio() reads (higher = long within that
sleeve). A sleeve score may blend several factors: rank them cross-sectionally (rank(axis=1, pct=True) - 0.5) and
take a weighted mean (weights as numeric params), or use weights estimated ONLY on past rows, or a purged
walk-forward model (Ridge / HistGradientBoosting, deterministic, refit at most monthly on labels that ended before
the fold start). portfolio() selects names by rank, so only the cross-sectional order of each sleeve score matters
unless portfolio() says otherwise. Keep it fast: the full evaluation reruns the spec ~15 times."""

TS_PORTFOLIO_GUIDE = """portfolio(S, F, bundle, params) gets the sleeve scores S (from combine) and the factors F and
returns the target positions (ts x symbol, same index/columns as close, NaN or 0 = flat, as the ts engine expects).
It owns the universe filter, names per side, cohort / clock schedule, sleeve budgets, gross scaling and risk
filters. Vectorised, trailing windows only."""


def is_ts_state(state: dict) -> bool:
    return bool(state.get("portfolio"))


def parent_engine(parent_code: str) -> str:
    return str(_literal_spec(parent_code).get("engine"))


def _literal_spec(parent_code: str) -> dict:
    for node in ast.parse(parent_code).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SPEC" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("parent has no literal SPEC dict")


def base_of(parent_code: str) -> tuple[str, dict, dict]:
    """(renamed parent module, SPEC without params, params): helpers stay importable as _parent_* names."""
    spec = _literal_spec(parent_code)
    base = _strip_future(parent_code)
    base = re.sub(r"^SPEC\s*=", "_PARENT_SPEC =", base, count=1, flags=re.M)
    base = re.sub(r"^def signal\(", "def _parent_signal(", base, count=1, flags=re.M)
    params = dict(spec.get("params") or {})
    return base, {k: v for k, v in spec.items() if k != "params"}, params


def split_functions(code: str) -> dict[str, str]:
    """Top-level functions of a code block by name (source text, decorators included); other statements dropped."""
    tree = ast.parse(code)
    lines = code.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start:node.end_lineno]) + "\n"
    return out


def _strip_future(code: str) -> str:
    return re.sub(r"^from __future__ import annotations\s*$", "", code, flags=re.M)


def seed_from_parent(parent_code: str) -> dict:
    """The parent signal becomes factor 'parent'; model = rank mean; portfolio = parent SPEC."""
    base, spec, params = base_of(parent_code)
    if spec.get("engine") != "xs":
        raise ValueError(f"seed_from_parent needs an xs parent (engine={spec.get('engine')}); "
                         "ts parents are split by decompose.decompose_ts")
    return {"base": base, "factors": {"parent": "def fac_parent(bundle, params):\n"
                                                "    return _parent_signal(bundle, params)\n"},
            "model": MODEL_RANK_MEAN, "spec": spec, "params": params}


def assemble(state: dict, stem: str, note: str = "") -> str:
    spec = dict(state["spec"])
    spec["id"] = stem
    spec["params"] = state["params"]
    names = list(state["factors"])
    spec["assumptions"] = (f"Factor+model composite (RD loop). Factors: {', '.join(names)}. {note} "
                           f"Base: {str(spec.get('assumptions', ''))[:600]}")[:2000]
    parts = ['"""Auto-assembled factor+model spec (rdagent zoo_fm)."""', "from __future__ import annotations", "",
             "import numpy as np", "import pandas as pd", "",
             "# ---- base (parent spec, renamed) " + "-" * 40, state["base"].strip(), "",
             "# ---- portfolio " + "-" * 58, "SPEC = " + pprint.pformat(spec, width=110, sort_dicts=False), "",
             *(["# ---- shared helpers " + "-" * 53, _strip_future(state["helpers"]).strip(), ""]
               if state.get("helpers") else []),
             "# ---- factors " + "-" * 60]
    for n in names:
        parts += [_strip_future(state["factors"][n]).strip(), ""]
    parts += [f"FACTORS = {{{', '.join(repr(n) + ': fac_' + n for n in names)}}}", "",
              "# ---- model " + "-" * 62, _strip_future(state["model"]).strip(), ""]
    if is_ts_state(state):
        parts += ["# ---- portfolio construction " + "-" * 45, _strip_future(state["portfolio"]).strip(), "",
                  "def signal(bundle, params):",
                  '    close = bundle.wide("close", "1h")',
                  "    F = {n: f(bundle, params).reindex_like(close) for n, f in FACTORS.items()}",
                  "    return portfolio(combine(F, bundle, params), F, bundle, params)", ""]
    else:
        parts += ["def signal(bundle, params):",
                  '    close = bundle.wide("close", "1h")',
                  "    F = {n: f(bundle, params).reindex_like(close) for n, f in FACTORS.items()}",
                  "    return combine(F, bundle, params).reindex_like(close).where(close.notna())", ""]
    return "\n".join(parts)


def factor_names(state: dict) -> list[str]:
    return list(state["factors"])


class ArmBandit:
    """Gaussian Thompson sampling over the three arms. Reward = score improvement over the best at proposal time
    (failed / disqualified runs = -0.5). Unlike the qlib bandit the reward is an improvement, not a level, so the
    bandit learns which kind of change tends to help."""

    def __init__(self, prior_mu: float = 0.0, prior_sd: float = 0.3, noise_sd: float = 0.3, seed: int | None = None):
        self.prior_mu, self.prior_sd, self.noise_sd = prior_mu, prior_sd, noise_sd
        self.obs: dict[str, list[float]] = {a: [] for a in ARMS}
        self.rng = random.Random(seed)

    def update(self, arm: str, reward: float) -> None:
        self.obs[arm].append(float(reward))

    def posterior(self, arm: str) -> tuple[float, float]:
        n = len(self.obs[arm])
        p0, pn = 1 / self.prior_sd ** 2, n / self.noise_sd ** 2
        mu = (self.prior_mu * p0 + (sum(self.obs[arm]) / self.noise_sd ** 2)) / (p0 + pn)
        return mu, math.sqrt(1 / (p0 + pn))

    def choose(self) -> str:
        draws = {a: self.rng.gauss(*self.posterior(a)) for a in ARMS}
        return max(draws, key=draws.get)

    def summary(self) -> str:
        return json.dumps({a: {"n": len(self.obs[a]), "mean": round(self.posterior(a)[0], 3)} for a in ARMS})
