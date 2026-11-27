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


def _fix_litellm_types() -> None:
    """litellm 1.97 + pydantic 2.14: Message/ModelResponse have unresolved forward refs
    (ChatCompletionReasoningSummaryTextBlock) and cannot be instantiated. Rebuild them once."""
    try:
        import litellm  # noqa: F401
        import litellm.types.llms.openai as lo
        import litellm.types.utils as tu
    except Exception:  # noqa: BLE001
        return
    ns = {**vars(lo), **vars(tu)}
    for n in ("Delta", "Message", "Choices", "StreamingChoices", "ModelResponse"):
        c = getattr(tu, n, None)
        if c is not None and hasattr(c, "model_rebuild"):
            try:
                c.model_rebuild(force=True, _types_namespace=ns)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"litellm type rebuild failed for {n}: {e}")


_fix_litellm_types()


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


# ---- shared dollar budget (all zoo jobs on this machine) ----------------------------------------------------------
class UsdBudgetExceeded(RuntimeError):
    pass


_LAST: dict = {}


def _litellm_cb(kwargs, completion_response, start_time, end_time):  # noqa: ARG001
    try:
        u = getattr(completion_response, "usage", None) or completion_response.get("usage")
        _LAST["in"] = int(getattr(u, "prompt_tokens", None) or u["prompt_tokens"])
        _LAST["out"] = int(getattr(u, "completion_tokens", None) or u["completion_tokens"])
    except Exception:  # noqa: BLE001
        pass


try:
    import litellm
    if _litellm_cb not in litellm.success_callback:
        litellm.success_callback.append(_litellm_cb)
except Exception:  # noqa: BLE001
    pass


def _usd_path() -> Path:
    p = Path(S.usage_file).expanduser() if S.usage_file else Path(S.work).expanduser() / "llm_usage_usd.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _usd_update(fn):
    """Read-modify-write the shared usage file under an exclusive lock."""
    import fcntl
    p = _usd_path()
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            d = json.loads(f.read() or "{}")
        except Exception:  # noqa: BLE001
            d = {}
        d = {"usd": 0.0, "input": 0, "output": 0, "calls": 0, "estimated_calls": 0, **d}
        out = fn(d)
        f.seek(0)
        f.truncate()
        f.write(json.dumps(d))
        fcntl.flock(f, fcntl.LOCK_UN)
    return out


def usd_used() -> float:
    return _usd_update(lambda d: d["usd"])


def _cost(i: int, o: int) -> float:
    return i * S.price_in / 1e6 + o * S.price_out / 1e6


def chat(user_prompt: str, system_prompt: str | None = None, former_messages: list | None = None,
         json_mode: bool = False) -> str:
    u = usage()
    inp = _est([system_prompt, former_messages, user_prompt])
    if S.token_budget and u["input"] + u["output"] + inp > S.token_budget:
        raise TokenBudgetExceeded(f"token budget {S.token_budget} would be exceeded (used {u})")
    reserve = _cost(inp, S.reserve_out_tokens)
    if S.usd_budget:
        def charge(d):  # reserve the worst case of this call before sending it
            if d["usd"] + reserve > S.usd_budget:
                raise UsdBudgetExceeded(f"USD budget {S.usd_budget} reached (used {d['usd']:.2f})")
            d["usd"] += reserve
        _usd_update(charge)
    kw = {"json_mode": True} if json_mode else {}
    # count the request before sending it, so failed calls (which may still be billed) are counted too
    u = {"input": u["input"] + inp, "output": u["output"], "calls": u["calls"] + 1}
    _path().write_text(json.dumps(u))
    _LAST.clear()
    resp = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt, system_prompt=system_prompt, former_messages=former_messages or [], **kw)
    real = "in" in _LAST
    ti, to = (_LAST["in"], _LAST["out"]) if real else (inp, _est(resp))
    u["output"] += _est(resp)
    _path().write_text(json.dumps(u))

    def settle(d):  # replace the reservation with the actual (or estimated) cost
        d["usd"] += _cost(ti, to) - (reserve if S.usd_budget else 0.0)
        d["input"] += ti
        d["output"] += to
        d["calls"] += 1
        d["estimated_calls"] += 0 if real else 1
        return d["usd"]
    total = _usd_update(settle)
    logger.info(f"zoo llm call: {ti} in / {to} out tokens ({'reported' if real else 'estimated'}), "
                f"total ${total:.3f} of ${S.usd_budget}")
    return resp
