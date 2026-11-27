#!/usr/bin/env bash
# Detached zoo_spec job: survives the ssh session, leaves a small status file that one short ssh read can check.
#   setsid nohup bash zoo_spec_job.sh <name> <parent> <loop_n> > /dev/null 2>&1 < /dev/null &
#   later:  cat ~/zoo_rd_work/jobs/<name>.status      (state=running|done|failed, exit code, times, token usage)
set -u
NAME=$1; PARENT=$2; LOOPS=${3:-1}
J=$HOME/zoo_rd_work/jobs; mkdir -p "$J"; ST=$J/$NAME.status; LOG=$J/$NAME.log
echo "state=running pid=$$ parent=$PARENT loops=$LOOPS start=$(date -u +%FT%TZ)" > "$ST"
source ~/rdagent-public-20261123/miniconda/bin/activate rdagent4qlib
set -a; source ~/.k3.env; set +a
export OPENAI_API_KEY=$K3_KEY OPENAI_API_BASE=${OPENAI_API_BASE:-https://cloud.infini-ai.com/maas/v1/} \
  CHAT_MODEL=${CHAT_MODEL:-openai/kimi-k3} CHAT_TEMPERATURE=${CHAT_TEMPERATURE:-1.0} MAX_RETRY=${MAX_RETRY:-2} \
  LITELLM_LOCAL_MODEL_COST_MAP=True ZOO_REPO=$HOME/quant-strategy-zoo ZOO_DATA=/datasets/data2/research/data \
  ZOO_WORK=$HOME/zoo_rd_work ZOO_PYTHON=$HOME/venvs/zoo/bin/python ZOO_EVAL_THREADS=${ZOO_EVAL_THREADS:-2} \
  OMP_NUM_THREADS=2 ZOO_TOKEN_BUDGET=${ZOO_TOKEN_BUDGET:-200000} WORKSPACE_PATH=$HOME/zoo_rd_work/ws/$NAME
unset K3_KEY
timeout ${JOB_TIMEOUT:-6h} rdagent zoo_spec --parent "$PARENT" --loop-n "$LOOPS" > "$LOG" 2>&1
RC=$?
USAGE=$(cat "$WORKSPACE_PATH/zoo_token_usage.json" 2>/dev/null || echo '{}')
[ $RC -eq 0 ] && S=done || S=failed
echo "state=$S exit=$RC parent=$PARENT loops=$LOOPS end=$(date -u +%FT%TZ) usage=$USAGE" >> "$ST"
