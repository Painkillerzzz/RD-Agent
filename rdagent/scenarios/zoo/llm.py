"""All LLM calls of the zoo scenario go through here, with a hard token budget.

Tokens are estimated from characters (~4 chars per token, both directions) because the backend returns text only;
the estimate is deliberately conservative. The running total is kept in <workspace>/zoo_token_usage.json so it
survives restarts; set ZOO_TOKEN_BUDGET (0 = unlimited)."""
from __future__ import annotations

import json
from pathlib import Path

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S


class TokenBudgetExceeded(RuntimeError):
    pass


def _path() -> Path:
    p = Path(RD_AGENT_SETTINGS.workspace_path) / "zoo_token_usage.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def usage() -> dict:
    try:
        return json.loads(_path().read_text())
    except Exception:  # noqa: BLE001
        return {"input": 0, "output": 0, "calls": 0}


def _est(x) -> int:
    return len(json.dumps(x, ensure_ascii=False)) // 3 + 1


def chat(user_prompt: str, system_prompt: str | None = None, former_messages: list | None = None,
         json_mode: bool = False) -> str:
    u = usage()
    inp = _est([system_prompt, former_messages, user_prompt])
    if S.token_budget and u["input"] + u["output"] + inp > S.token_budget:
        raise TokenBudgetExceeded(f"token budget {S.token_budget} would be exceeded (used {u})")
    kw = {"json_mode": True} if json_mode else {}
    resp = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt, system_prompt=system_prompt, former_messages=former_messages or [], **kw)
    u = {"input": u["input"] + inp, "output": u["output"] + _est(resp), "calls": u["calls"] + 1}
    _path().write_text(json.dumps(u))
    logger.info(f"zoo llm usage (estimated tokens): {u}")
    return resp
