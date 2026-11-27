from __future__ import annotations

import json

from rdagent.core.proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen, Trace
from rdagent.scenarios.zoo import llm
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S
from rdagent.scenarios.zoo.experiment import ZooSpecExperiment, ZooSpecTask

SYSTEM = """You are a senior crypto quant researcher. You propose ONE concrete, testable modification of the current
best strategy spec per iteration. Respond with a JSON object with keys: hypothesis (what to change, concretely),
reason (market mechanism and why it should raise the DEV objective), concise_reason, concise_observation (what the
history shows), concise_justification, concise_knowledge (a reusable lesson, "if ... then ...")."""


def sota(trace: Trace):
    """Best accepted experiment so far, or None (then the parent spec is the SOTA)."""
    best, best_s = None, None
    for exp, fb in trace.hist:
        s = ev.score(getattr(exp, "result", None))
        if fb.decision and s is not None and (best_s is None or s > best_s):
            best, best_s = exp, s
    return best


def sota_code_and_view(trace: Trace, scen) -> tuple[str, dict | None]:
    b = sota(trace)
    if b is not None:
        return b.experiment_workspace.spec_code, b.result
    return scen.parent_code, getattr(scen, "parent_view", None)


def history_text(trace: Trace, n: int) -> str:
    rows = []
    for i, (exp, fb) in enumerate(trace.hist[-n:]):
        h = exp.hypothesis.hypothesis if exp.hypothesis else ""
        rows.append(f"- iteration {i}: hypothesis: {h}\n  result: {ev.brief(getattr(exp, 'result', None))}\n"
                    f"  accepted: {fb.decision}; reason: {fb.reason[:600]}")
    return "\n".join(rows) or "(no iterations yet)"


class ZooHypothesisGen(HypothesisGen):
    def gen(self, trace: Trace, plan=None) -> Hypothesis:  # noqa: ARG002
        code, view = sota_code_and_view(trace, self.scen)
        user = (f"{self.scen.get_scenario_all_desc()}\n\n# Current best spec\n```python\n{code}\n```\n"
                f"# Its evaluation\n{ev.brief(view)}\n\n# Recent iterations\n{history_text(trace, S.history_n)}\n\n"
                "Propose the next modification.")
        resp = llm.chat(user, SYSTEM, json_mode=True)
        d = json.loads(resp)
        return Hypothesis(hypothesis=str(d.get("hypothesis", "")), reason=str(d.get("reason", "")),
                          concise_reason=str(d.get("concise_reason", "")),
                          concise_observation=str(d.get("concise_observation", "")),
                          concise_justification=str(d.get("concise_justification", "")),
                          concise_knowledge=str(d.get("concise_knowledge", "")))


class ZooHypothesis2Experiment(Hypothesis2Experiment[ZooSpecExperiment]):
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ZooSpecExperiment:
        code, _ = sota_code_and_view(trace, trace.scen)
        stem = f"rd{len(trace.hist):03d}-{S.parent}"[:120]
        task = ZooSpecTask(name=stem, description=hypothesis.hypothesis, base_code=code, stem=stem)
        exp = ZooSpecExperiment(sub_tasks=[task], hypothesis=hypothesis)
        return exp
