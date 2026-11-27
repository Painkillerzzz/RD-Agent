#!/usr/bin/env bash
# Continuous optimisation of one zoo strategy until the shared USD budget is used up.
#   ZOO_USD_BUDGET=50 setsid nohup bash zoo_continuous.sh <name> <parent> <zoo_fm|zoo_spec> > /dev/null 2>&1 < /dev/null &
# Status: ~/zoo_rd_work/jobs/<name>.status (one line per (re)start/exit). Resumes the RD-Agent session after any crash
# (LLM hiccup, server restart of the job), stops when the budget is reached or after 3 quick consecutive failures.
set -u
NAME=$1; PARENT=$2; CMD=${3:-zoo_fm}
J=$HOME/zoo_rd_work/jobs; R=$HOME/zoo_rd_work/runs/$NAME; mkdir -p "$J" "$R"
ST=$J/$NAME.status; LOG=$J/$NAME.log
source ~/rdagent-public-20261123/miniconda/bin/activate rdagent4qlib
set -a; source ~/.k3.env; set +a
export OPENAI_API_KEY=$K3_KEY OPENAI_API_BASE=${OPENAI_API_BASE:-https://cloud.infini-ai.com/maas/v1/} \
  CHAT_MODEL=${CHAT_MODEL:-openai/kimi-k3} CHAT_TEMPERATURE=${CHAT_TEMPERATURE:-1.0} MAX_RETRY=${MAX_RETRY:-2} \
  CHAT_STREAM=False LITELLM_LOCAL_MODEL_COST_MAP=True ZOO_REPO=$HOME/quant-strategy-zoo \
  ZOO_DATA=/datasets/data2/research/data ZOO_WORK=$HOME/zoo_rd_work ZOO_PYTHON=$HOME/venvs/zoo/bin/python \
  ZOO_EVAL_THREADS=${ZOO_EVAL_THREADS:-2} OMP_NUM_THREADS=2 ZOO_TOKEN_BUDGET=0 \
  ZOO_USD_BUDGET=${ZOO_USD_BUDGET:?set ZOO_USD_BUDGET} ZOO_VAL_ROUND=${ZOO_VAL_ROUND:-$NAME} \
  WORKSPACE_PATH=$R/ws LOG_TRACE_PATH=$R/log
unset K3_KEY
usd() { python - <<'PY'
import json, os
try: print(json.load(open(os.path.expanduser("~/zoo_rd_work/llm_usage_usd.json")))["usd"])
except Exception: print(0)
PY
}
fails=0
while :; do
  U=$(usd)
  if python -c "import sys; sys.exit(0 if float('$U') + 0.5 >= float('$ZOO_USD_BUDGET') else 1)"; then
    echo "state=done reason=budget usd=$U end=$(date -u +%FT%TZ)" >> "$ST"; break
  fi
  ARGS=(--parent "$PARENT" --loop-n 100000)
  [ -d "$R/log/__session__" ] && ARGS+=(--path "$R/log/__session__")
  echo "state=running parent=$PARENT cmd=$CMD usd=$U resume=$([ -d "$R/log/__session__" ] && echo yes || echo no) start=$(date -u +%FT%TZ)" >> "$ST"
  t0=$(date +%s)
  # --foreground keeps rdagent in this script's process group, so kill -TERM -- -<driver pgid> stops everything
  timeout --foreground ${JOB_TIMEOUT:-48h} rdagent "$CMD" "${ARGS[@]}" >> "$LOG" 2>&1
  RC=$?; dt=$(( $(date +%s) - t0 ))
  echo "state=exited exit=$RC secs=$dt usd=$(usd) at=$(date -u +%FT%TZ)" >> "$ST"
  if [ $dt -lt 300 ]; then fails=$((fails+1)); else fails=0; fi
  if [ $fails -ge 3 ]; then echo "state=failed reason=3_quick_failures at=$(date -u +%FT%TZ)" >> "$ST"; break; fi
  sleep 120
done
