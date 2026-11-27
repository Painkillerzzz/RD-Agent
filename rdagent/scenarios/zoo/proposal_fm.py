"""Factor + model + portfolio loop: a bandit picks the arm, the LLM proposes one change for that arm."""
from __future__ import annotations

import copy
import json

from rdagent.core.proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo import fm, llm
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S
from rdagent.scenarios.zoo.experiment import ZooSpecExperiment, ZooSpecTask

ARM_TASK = {
    "factor": "Propose ONE new factor for the library (a ts x symbol score, higher = expected to outperform "
              "cross-sectionally over the holding period), or replace the weakest non-parent factor if the library "
              f"has {fm.MAX_FACTORS}. It must be economically different from the existing factors.",
    "model": "Propose ONE change of how the factors are combined into the final score. " + fm.MODEL_GUIDE,
    "portfolio": "Propose ONE change of portfolio construction only: SPEC keys rebalance, holding_bars, n_groups, "
                 "weighting, or universe/liquidity params already in params. No factor or model change. Note the "
                 "engine runs round(holding_bars / rebalance step) overlapping cohorts: with rebalance '1d' on 1h bars "
                 "holding_bars 12 and 24 are both one cohort; 48 = two overlapping daily books.",
}

SYSTEM = """You are a senior crypto quant researcher running a factor + model research loop. Each iteration changes
exactly one layer (the one requested). Respond with a JSON object with keys: hypothesis (the concrete change),
reason (mechanism and why it should raise the DEV objective), concise_reason, concise_observation,
concise_justification, concise_knowledge (a reusable "if ... then ..." lesson). For the portfolio layer also give
"spec_updates" (dict of SPEC keys) and "param_updates" (dict of params); for the factor layer give "factor_name"
(snake_case) and optionally "replace" (an existing factor name)."""


def best(trace: Trace, scen):
    """(state, view) of the latest accepted experiment (each had to beat its parent), else the seeded parent."""
    for exp, fb in reversed(trace.hist):
        if fb.decision:
            return exp.fm_state, exp.result
    return scen.fm_seed, getattr(scen, "parent_view", None)


def bandit_from_trace(trace: Trace) -> fm.ArmBandit:
    """Replays the history. Reward = DEV objective - best DEV objective at proposal time, +0.5 if the gate accepted;
    failed / not evaluable = -0.5."""
    bd = fm.ArmBandit(seed=len(trace.hist))
    for exp, fb in trace.hist:
        arm = getattr(exp, "arm", None)
        if arm in fm.ARMS:
            s, b = ev.score(getattr(exp, "result", None)), getattr(exp, "best_score_before", None)
            r = -0.5 if s is None else s - (b or 0.0)
            bd.update(arm, r + (0.5 if fb.decision else 0.0))
    return bd


def history_text(trace: Trace, n: int) -> str:
    rows = []
    for i, (exp, fb) in enumerate(trace.hist[-n:]):
        h = exp.hypothesis.hypothesis if exp.hypothesis else ""
        rows.append(f"- [{getattr(exp, 'arm', '?')}] {h}\n  result: {ev.brief(getattr(exp, 'result', None))}\n"
                    f"  accepted: {fb.decision}; reason: {fb.reason[:400]}")
    return "\n".join(rows) or "(no iterations yet)"


def state_text(state: dict) -> str:
    fac = "\n\n".join(state["factors"].values())
    return (f"## Portfolio (SPEC without params)\n{json.dumps(state['spec'], default=str)[:1500]}\n"
            f"## params\n{json.dumps(state['params'])}\n## Factor library\n```python\n{fac}\n```\n"
            f"(fac_parent calls the parent spec's signal, shown below as _parent_signal)\n"
            f"## Model\n```python\n{state['model']}\n```")


class ZooFMHypothesisGen(HypothesisGen):
    def gen(self, trace: Trace, plan=None) -> Hypothesis:  # noqa: ARG002
        state, view = best(trace, self.scen)
        bd = bandit_from_trace(trace)
        arm = "factor" if not trace.hist else bd.choose()
        logger.info(f"zoo_fm bandit {bd.summary()} -> {arm}")
        user = (f"{self.scen.get_scenario_all_desc()}\n\n# Parent spec (base)\n```python\n{self.scen.parent_code}\n```\n"
                f"# Current best decomposition\n{state_text(state)}\n# Its evaluation\n{ev.brief(view)}\n\n"
                f"# Recent iterations\n{history_text(trace, S.history_n)}\n\n# Layer to change now: {arm}\n"
                f"{ARM_TASK[arm]}")
        d = json.loads(llm.chat(user, SYSTEM, json_mode=True))
        h = Hypothesis(hypothesis=str(d.get("hypothesis", "")), reason=str(d.get("reason", "")),
                       concise_reason=str(d.get("concise_reason", "")),
                       concise_observation=str(d.get("concise_observation", "")),
                       concise_justification=str(d.get("concise_justification", "")),
                       concise_knowledge=str(d.get("concise_knowledge", "")))
        h.arm, h.extra = arm, d
        return h


class ZooFMHypothesis2Experiment(Hypothesis2Experiment[ZooSpecExperiment]):
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ZooSpecExperiment:
        state, view = best(trace, trace.scen)
        stem = f"fm{len(trace.hist):03d}-{S.parent}"[:120]
        task = ZooSpecTask(name=stem, description=hypothesis.hypothesis, base_code="", stem=stem)
        exp = ZooSpecExperiment(sub_tasks=[task], hypothesis=hypothesis)
        exp.arm = getattr(hypothesis, "arm", "factor")
        exp.fm_extra = getattr(hypothesis, "extra", {}) or {}
        exp.fm_base_state = copy.deepcopy(state)
        exp.fm_state = copy.deepcopy(state)
        exp.best_score_before = ev.score(view)
        return exp
