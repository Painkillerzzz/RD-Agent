from __future__ import annotations

import re
from pathlib import Path

from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, RunnerError
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S
from rdagent.scenarios.zoo.experiment import ZooSpecExperiment

CODER_SYSTEM = """You are an expert Python quant developer. You rewrite a zoo_val strategy spec module to implement a
given hypothesis. Return the COMPLETE new module in one ```python block and nothing else. Keep the SPEC dict
(update params, assumptions, leakage_notes and SPEC['id'] to the new stem), keep `signal(bundle, params)`.
Vectorised pandas/numpy, no network, no file I/O, no prints. Never use future rows."""


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return (max(m, key=len) if m else text).strip() + "\n"


class ZooSpecCoder(Developer[ZooSpecExperiment]):
    def develop(self, exp: ZooSpecExperiment) -> ZooSpecExperiment:
        task = exp.sub_tasks[0]
        ws = exp.experiment_workspace
        ws.prepare()
        spec_path = Path(ws.workspace_path) / f"{task.stem}.py"
        msgs: list = []
        prompt = (f"{self.scen.get_scenario_all_desc()}\n\n# Current spec\n```python\n{task.base_code}\n```\n\n"
                  f"# Hypothesis to implement\n{task.description}\n\nNew SPEC['id']: {task.stem}")
        log = ""
        for i in range(S.coder_max_tries):
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt=prompt, system_prompt=CODER_SYSTEM,
                                                                          former_messages=msgs)
            code = extract_code(resp)
            spec_path.write_text(code)
            ws.inject_files(**{ws.SPEC_FILE: code})
            ok, log = ev.check(spec_path)
            logger.info(f"zoo coder try {i + 1}: check ok={ok}")
            if ok:
                exp.check_log = log
                exp.sub_workspace_list = [ws]
                return exp
            msgs += [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
            prompt = ("zoo_val.check failed. Fix the module and return the complete corrected file.\n"
                      f"```\n{log[-4000:]}\n```")
        exp.check_log = log
        raise CoderError(f"spec did not pass zoo_val.check after {S.coder_max_tries} tries:\n{log[-2000:]}")


class ZooSpecRunner(Developer[ZooSpecExperiment]):
    def develop(self, exp: ZooSpecExperiment) -> ZooSpecExperiment:
        task = exp.sub_tasks[0]
        spec_path = Path(exp.experiment_workspace.workspace_path) / f"{task.stem}.py"
        view = ev.evaluate(spec_path, self.scen.family, note=task.description, parent=S.parent)
        exp.result = view
        logger.info(f"zoo runner {task.stem}: {ev.brief(view)}")
        if view.get("status") != "ok":
            raise RunnerError(f"evaluation {view.get('status')}: {ev.brief(view)}")
        return exp
