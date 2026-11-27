"""RD loop that optimises one quant-strategy-zoo spec against the local devval evaluator.

    rdagent zoo_spec --parent r0100-crypto_hourly_institutional_footprints_ensemble --loop-n 10

Paths and limits come from ZOO_* environment variables (see rdagent/scenarios/zoo/conf.py).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fire

from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError, RunnerError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING, ZooFMPropSetting


class ZooRDLoop(RDLoop):
    skip_loop_error = (CoderError, RunnerError)
    skip_loop_error_stepname = "feedback"

    def __init__(self, setting=ZOO_PROP_SETTING):
        super().__init__(setting)
        scen = self.trace.scen
        try:  # baseline already evaluated (cached on the evaluation machine)
            scen.parent_view = json.loads(ev.read_text(ev.WORK / "agent_views" / f"{setting.parent}.json"))
        except Exception:  # noqa: BLE001
            spec = Path(RD_AGENT_SETTINGS.workspace_path) / "zoo_parents" / f"{setting.parent}.py"
            spec.parent.mkdir(parents=True, exist_ok=True)
            spec.write_text(scen.parent_code)
            logger.info(f"evaluating parent {setting.parent} (baseline)")
            scen.parent_view = ev.evaluate(spec, scen.family, note="baseline: parent spec unchanged")
        logger.info(f"parent baseline: {ev.brief(scen.parent_view)}")
        if isinstance(setting, ZooFMPropSetting):
            from rdagent.scenarios.zoo import fm
            scen.fm_seed = fm.seed_from_parent(scen.parent_code)


def main(parent: str | None = None, path=None, step_n: int | None = None, loop_n: int | None = None,
         all_duration: str | None = None, checkout: bool = True):
    if parent:
        ZOO_PROP_SETTING.parent = parent
    loop = ZooRDLoop(ZOO_PROP_SETTING) if path is None else ZooRDLoop.load(path, checkout=checkout)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


def main_fm(parent: str | None = None, path=None, step_n: int | None = None, loop_n: int | None = None,
            all_duration: str | None = None, checkout: bool = True):
    """Factor + model + portfolio loop: a Thompson bandit picks the layer each iteration."""
    setting = ZooFMPropSetting()
    if parent:
        setting.parent = parent
    ZOO_PROP_SETTING.parent = setting.parent  # modules read ZOO_PROP_SETTING
    loop = ZooRDLoop(setting) if path is None else ZooRDLoop.load(path, checkout=checkout)
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
