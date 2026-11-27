from __future__ import annotations

import json

from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo import gate, llm
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S

SYSTEM = """You review one iteration of a strategy optimisation loop. The accept/reject decision was already made by
the zoo optimisation gate (DEV margin, then a blind VAL check; you only get pass/fail and coarse reasons, never VAL
numbers - do not speculate about VAL values). Explain the DEV result and suggest the next step. Respond with JSON
keys: observations, hypothesis_evaluation, new_hypothesis (a suggestion for the next iteration), reason (two
sentences)."""


def best_stem(trace: Trace) -> str:
    """Latest accepted candidate (each acceptance had to beat its parent), else the zoo parent."""
    for exp, fb in reversed(trace.hist):
        if fb.decision:
            return exp.sub_tasks[0].stem
    return S.parent


def best_view(trace: Trace, scen) -> dict | None:
    for exp, fb in reversed(trace.hist):
        if fb.decision:
            return exp.result
    return getattr(scen, "parent_view", None)


def gate_text(g: dict | None) -> str:
    if not g:
        return "gate: not run"
    return f"gate: {g.get('status')} reasons={g.get('reasons')} flags={g.get('flags')}"


class ZooFeedback(Experiment2Feedback):
    def generate_feedback(self, exp, trace: Trace, exception=None) -> HypothesisFeedback:
        stem, parent_stem = exp.sub_tasks[0].stem, best_stem(trace)
        cand, par, orig = gate.sealed(stem), gate.sealed(parent_stem), gate.sealed(S.parent)
        if exp.result is None or (exp.result or {}).get("status") != "ok" or not (cand and par and orig):
            g = {"status": "fail", "pass": False, "reasons": ["not_evaluated"], "flags": []}
        else:
            rid = f"{self.scen.val_round}-b{len(trace.hist) // max(S.val_round_iters, 1)}"
            try:
                g = gate.decide(stem, cand, par, orig, rid)
            except gate.ValBudgetExhausted:  # no VAL information this time; keep exploring on DEV
                g = {"status": "fail", "pass": False, "reasons": ["val_budget_round"], "flags": ["val_not_queried"]}
        exp.gate = g
        decision = bool(g.get("pass"))
        if decision:  # keep every accepted spec (code + gate verdict, no numbers) for review
            try:
                from pathlib import Path
                d = Path(S.work).expanduser() / "accepted" / S.parent
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{stem}.py").write_text(exp.experiment_workspace.spec_code)
                (d / f"{stem}.gate.json").write_text(json.dumps(g))
            except Exception:  # noqa: BLE001
                pass
        user = (f"Hypothesis: {exp.hypothesis.hypothesis if exp.hypothesis else ''}\n"
                f"New result: {ev.brief(exp.result)}\nCurrent best ({parent_stem}): {ev.brief(best_view(trace, self.scen))}\n"
                f"Decision: {'ACCEPT' if decision else 'REJECT'}; {gate_text(g)}")
        try:
            d = json.loads(llm.chat(user, SYSTEM, json_mode=True))
        except (llm.TokenBudgetExceeded, llm.UsdBudgetExceeded):
            raise
        except Exception as e:  # noqa: BLE001
            d = {"reason": f"feedback LLM failed: {e}"}
        return HypothesisFeedback(reason=f"{gate_text(g)}. {d.get('reason', '')}", decision=decision,
                                  observations=str(d.get("observations", "")),
                                  hypothesis_evaluation=str(d.get("hypothesis_evaluation", "")),
                                  new_hypothesis=str(d.get("new_hypothesis", "")), acceptable=decision)
