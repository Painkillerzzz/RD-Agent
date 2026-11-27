"""Settings for the quant-strategy-zoo spec optimisation loop (env prefix ZOO_)."""
from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class ZooPropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="ZOO_", protected_namespaces=())

    scen: str = "rdagent.scenarios.zoo.scenario.ZooSpecScenario"
    hypothesis_gen: str = "rdagent.scenarios.zoo.proposal.ZooHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.zoo.proposal.ZooHypothesis2Experiment"
    coder: str = "rdagent.scenarios.zoo.developer.ZooSpecCoder"
    runner: str = "rdagent.scenarios.zoo.developer.ZooSpecRunner"
    summarizer: str = "rdagent.scenarios.zoo.feedback.ZooFeedback"

    # where things live
    repo: str = "~/quant-strategy-zoo"
    """quant-strategy-zoo checkout (needs validation/ and blackbox/local/local_eval.py)"""
    data: str = "/datasets/data2/research/data"
    """research view cut at 2024-12-31 (DEV+VAL). Never the full store: TEST/FORWARD must stay sealed."""
    work: str = "~/zoo_rd_work"
    """local evaluator ledger/results and generated specs"""
    python: str = "~/venvs/zoo/bin/python"
    """python with pandas/pyarrow/scipy/sklearn used to run zoo_val and the evaluator"""

    # remote mode: run check/evaluation on another machine over plain ssh (no port forwarding needed)
    ssh: str = ""
    """e.g. "-p 22472 -o KexAlgorithms=curve25519-sha256 -o HostKeyAlgorithms=ssh-ed25519 painkillerzzz@10.213.74.1".
    When set, repo/data/work/python are paths ON THAT MACHINE and only the LLM side runs locally."""

    # what to optimise
    parent: str = "r0100-crypto_hourly_institutional_footprints_ensemble"
    """stem of the starting spec in research_results/specs/"""
    family: str = ""
    """evaluator family; empty = take it from research_results/specs.csv for the parent"""

    coder_max_tries: int = 4
    """LLM rewrite attempts until zoo_val.check passes"""
    eval_timeout: int = 7200
    token_budget: int = 200_000
    """hard cap on estimated LLM tokens (input + output) for this workspace; 0 = unlimited"""
    history_n: int = 8
    """how many past loops are shown to the planner"""


ZOO_PROP_SETTING = ZooPropSetting()
