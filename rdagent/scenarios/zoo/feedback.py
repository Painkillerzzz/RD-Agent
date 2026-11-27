from __future__ import annotations

import json

from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.scenarios.zoo import llm
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.proposal import sota_code_and_view

SYSTEM = """You review one iteration of a strategy optimisation loop. The accept/reject decision is already made by
a fixed rule (DEV objective vs the current best). Explain the result. Respond with JSON keys: observations,
hypothesis_evaluation, new_hypothesis (a suggestion for the next iteration), reason (two sentences)."""


class ZooFeedback(Experiment2Feedback):
    def generate_feedback(self, exp, trace: Trace, exception=None) -> HypothesisFeedback:
        _, best_view = sota_code_and_view(trace, self.scen)
        new_s, best_s = ev.score(exp.result), ev.score(best_view)
        decision = new_s is not None and (best_s is None or new_s > best_s + 0.02)
        user = (f"Hypothesis: {exp.hypothesis.hypothesis if exp.hypothesis else ''}\n"
                f"New result: {ev.brief(exp.result)}\nCurrent best: {ev.brief(best_view)}\n"
                f"Rule decision: {'ACCEPT' if decision else 'REJECT'} (new score {new_s}, best score {best_s})")
        try:
            d = json.loads(llm.chat(user, SYSTEM, json_mode=True))
        except llm.TokenBudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            d = {"reason": f"feedback LLM failed: {e}"}
        return HypothesisFeedback(reason=str(d.get("reason", "")), decision=decision,
                                  observations=str(d.get("observations", "")),
                                  hypothesis_evaluation=str(d.get("hypothesis_evaluation", "")),
                                  new_hypothesis=str(d.get("new_hypothesis", "")), acceptable=decision)
