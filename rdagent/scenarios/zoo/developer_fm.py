"""Coder for the factor + model loop: writes one function (factor or combine) or applies portfolio updates,
assembles the full spec and retries until zoo_val.check passes."""
from __future__ import annotations

import re
from pathlib import Path

from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo import fm, llm
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S
from rdagent.scenarios.zoo.developer import extract_code
from rdagent.scenarios.zoo.experiment import ZooSpecExperiment
from rdagent.scenarios.zoo.proposal_fm import state_text

CODER_SYSTEM = """You are an expert Python quant developer. Return exactly ONE ```python block that defines the
requested function (helpers prefixed with an underscore are allowed in the same block) and, if the function needs
new tuning knobs, a second ```json block with a dict of NEW params (numeric, prefixed with the factor name).
Vectorised pandas/numpy (sklearn allowed for the model), no network, no file I/O, no prints, no look-ahead:
every rolling window ends at the current row, no shift(-k) except purged training labels inside combine()."""

SAFE = re.compile(r"[^a-z0-9_]")


def _json_block(text: str) -> dict:
    import json
    m = re.findall(r"```json\n(.*?)```", text, re.S)
    try:
        d = json.loads(m[-1]) if m else {}
        return {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except Exception:  # noqa: BLE001
        return {}


class ZooFMCoder(Developer[ZooSpecExperiment]):
    def develop(self, exp: ZooSpecExperiment) -> ZooSpecExperiment:
        task, arm, extra = exp.sub_tasks[0], exp.arm, exp.fm_extra
        ws = exp.experiment_workspace
        ws.prepare()
        spec_path = Path(ws.workspace_path) / f"{task.stem}.py"
        base = exp.fm_base_state

        ts = fm.is_ts_state(base)
        if arm == "portfolio" and not ts:
            st = {**base, "spec": dict(base["spec"]), "params": dict(base["params"])}
            for k, v in (extra.get("spec_updates") or {}).items():
                if k in ("rebalance", "holding_bars", "n_groups", "weighting", "direction"):
                    st["spec"][k] = v
            for k, v in (extra.get("param_updates") or {}).items():
                if k in st["params"] and isinstance(v, (int, float)) and not isinstance(v, bool):
                    st["params"][k] = v
            return self._finish(exp, st, spec_path, ws, [])

        if arm == "factor":
            name = SAFE.sub("_", str(extra.get("factor_name") or f"f{len(base['factors'])}").lower())[:40] or "f"
            if name == "parent" or name in base["factors"]:
                name = f"{name}_{len(base['factors'])}"
            want = f"def fac_{name}(bundle, params) -> pd.DataFrame  (ts x symbol, same index/columns as close)"
            if ts:
                want += ("\nand, in the same block, the full updated def combine(F, bundle, params) -> dict that "
                         f"blends fac_{name} into the sleeve(s) named in the change (" + fm.TS_MODEL_GUIDE + ")")
        elif arm == "model":
            want = ("def combine(F, bundle, params) -> dict  (" + fm.TS_MODEL_GUIDE + ")" if ts else
                    "def combine(F, bundle, params) -> pd.DataFrame  (" + fm.MODEL_GUIDE + ")")
        else:  # ts portfolio: rewritten as code, numeric params may change too
            want = "def portfolio(S, F, bundle, params) -> pd.DataFrame  (" + fm.TS_PORTFOLIO_GUIDE + ")"
        prompt = (f"{self.scen.get_scenario_all_desc(simple_background=True)}\n\n# Current decomposition\n"
                  f"{state_text(base)}\n\n# Change to implement ({arm})\n{task.description}\n\n# Write\n{want}")
        msgs: list = []
        log = ""
        for i in range(S.coder_max_tries):
            resp = llm.chat(prompt, CODER_SYSTEM, former_messages=msgs)
            code, newp = extract_code(resp), _json_block(resp)
            if arm == "factor":  # the LLM may name the function differently: force the assigned name
                code = re.sub(r"^def fac_\w+\(", f"def fac_{name}(", code, count=1, flags=re.M)
            st = {**base, "factors": dict(base["factors"]), "params": {**base["params"], **newp}}
            if ts and arm == "portfolio":
                for k, v in (extra.get("param_updates") or {}).items():
                    if k in st["params"] and isinstance(v, (int, float)) and not isinstance(v, bool):
                        st["params"][k] = v
            fns = {}
            if ts:
                try:
                    fns = fm.split_functions(code)
                except SyntaxError:
                    fns = {}
            if ts and arm == "factor" and "combine" in fns:  # the factor block also carries the rewired combine
                st["model"] = fns.pop("combine")
                code = "".join(fns.values())
            if arm == "portfolio" and ts:
                st["portfolio"] = code
            elif arm == "factor":
                rep = extra.get("replace")
                if rep in st["factors"] and rep != "parent":
                    st["factors"].pop(rep)
                elif len(st["factors"]) >= fm.MAX_FACTORS:
                    st["factors"].pop([n for n in st["factors"] if n != "parent"][0])
                st["factors"][name] = code
            elif arm == "model":
                st["model"] = code
            ok, log = self._check(st, task.stem, spec_path, ws)
            logger.info(f"zoo_fm coder [{arm}] try {i + 1}: check ok={ok}")
            if ok:
                exp.fm_state, exp.check_log = st, log
                exp.sub_workspace_list = [ws]
                return exp
            msgs += [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
            prompt = ("The assembled spec failed zoo_val.check. Fix your function and return it again in full.\n"
                      f"```\n{log[-4000:]}\n```")
        exp.check_log = log
        raise CoderError(f"[{arm}] did not pass zoo_val.check after {S.coder_max_tries} tries:\n{log[-2000:]}")

    def _check(self, st, stem, spec_path, ws):
        code = fm.assemble(st, stem, note=f"Layer changed: {stem}.")
        spec_path.write_text(code)
        ws.inject_files(**{ws.SPEC_FILE: code})
        try:
            compile(code, str(spec_path), "exec")
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        return ev.check(spec_path)

    def _finish(self, exp, st, spec_path, ws, _):
        ok, log = self._check(st, exp.sub_tasks[0].stem, spec_path, ws)
        exp.check_log = log
        if not ok:
            raise CoderError(f"[portfolio] assembled spec failed zoo_val.check:\n{log[-2000:]}")
        exp.fm_state = st
        exp.sub_workspace_list = [ws]
        return exp
