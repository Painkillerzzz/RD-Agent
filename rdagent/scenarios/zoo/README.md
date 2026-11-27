# zoo scenario: optimise a quant-strategy-zoo spec

`rdagent zoo_spec --parent <stem> --loop-n N` runs the usual RD loop on one zoo_val spec
(`research_results/specs/<stem>.py` in quant-strategy-zoo):

| step | class | what it does |
| --- | --- | --- |
| propose | `proposal.ZooHypothesisGen` | LLM proposes one modification of the current best spec, given the contract, the spec, its DEV metrics / VAL bucket and recent iterations |
| exp_gen | `proposal.ZooHypothesis2Experiment` | one task: rewrite the current best spec as `rdNNN-<parent>` |
| coding | `developer.ZooSpecCoder` | LLM writes the full module; `zoo_val.check` (schema, shape, causal) must pass, up to `ZOO_CODER_MAX_TRIES` rewrites with the error fed back |
| running | `developer.ZooSpecRunner` | `blackbox/local/local_eval.py` (same code as the server evaluator): real-panel causal check, DEV, VAL, 2x cost, +-20% params |
| feedback | `feedback.ZooFeedback` | fixed rule decides: accept if score beats the current best by 0.02; LLM explains |

Score (DEV only) = 0.5 * DEV Sharpe + 0.5 * min DEV Sharpe over parameter perturbations; requires DEV t >= 2 and a VAL
bucket other than `fail`. The agent never sees VAL numbers, only the bucket and gate booleans, as on the server.

Settings: `ZOO_REPO`, `ZOO_DATA` (research view cut at 2024-12-31), `ZOO_WORK`, `ZOO_PYTHON` (python with the zoo
deps), `ZOO_PARENT`, `ZOO_FAMILY`, `ZOO_CODER_MAX_TRIES`, `ZOO_EVAL_TIMEOUT`, `ZOO_HISTORY_N`. Runbook for the
training server: `tools/rdagent/SERVER.md` in quant-strategy-zoo.

Remote mode: set `ZOO_SSH` (ssh arguments, e.g. `-p 22472 user@host`) to keep the LLM side on this machine and run
`zoo_val.check` and the evaluator on the training server over plain ssh commands (no port forwarding). `ZOO_REPO`,
`ZOO_DATA`, `ZOO_WORK`, `ZOO_PYTHON` are then paths on the server.

## Running with kimi-k3 on infini-ai (tested 2026-11-27)

```bash
source ~/rdagent-public-20261123/miniconda/bin/activate rdagent4qlib
set -a; source ~/.k3.env; set +a
export OPENAI_API_KEY=$K3_KEY OPENAI_API_BASE=https://cloud.infini-ai.com/maas/v1/ CHAT_MODEL=openai/kimi-k3 \
  CHAT_TEMPERATURE=1.0 `# kimi-k3 thinking mode rejects any other temperature (HTTP 400)` \
  MAX_RETRY=2 LITELLM_LOCAL_MODEL_COST_MAP=True \
  ZOO_REPO=$HOME/quant-strategy-zoo ZOO_DATA=/datasets/data2/research/data ZOO_WORK=$HOME/zoo_rd_work \
  ZOO_PYTHON=$HOME/venvs/zoo/bin/python ZOO_EVAL_THREADS=2 OMP_NUM_THREADS=2 \
  ZOO_TOKEN_BUDGET=200000 WORKSPACE_PATH=$HOME/zoo_rd_work/ws
rdagent zoo_spec --parent r0120-c-crypto_hourly_absorbed_flow_reversal --loop-n 1
```

Estimated token usage accumulates in `$WORKSPACE_PATH/zoo_token_usage.json` (delete it to reset the budget).
One kimi-k3 round on r0120-c used ~9.5k input + 2.5k output tokens (~$0.07 at $3/$15 per 1M) plus ~17 min of evaluation.

## Submitting jobs to the training server (do this, not a raw nohup)

Use `bin/zoo_spec_job.sh`: it detaches from ssh, has a hard timeout, and writes `~/zoo_rd_work/jobs/<name>.status`
(`state=running`, then `state=done|failed exit=… usage=…`). Whoever launches a job must also schedule its own check
at the expected finish time (eval ≈ 17 min per round) and read only the status file plus a log tail in one short ssh
call. Never keep an ssh connection open to wait, and never rely on a session "remembering" to look later.

## Factor + model loop (`rdagent zoo_fm`, xs parents only)

Keeps RD-Agent(Q)'s factor/model co-evolution but evaluates with the zoo evaluator. The parent signal becomes factor
`parent`; the model starts as an equal-weight rank mean; the portfolio is the parent SPEC. Each iteration a Gaussian
Thompson bandit (`fm.ArmBandit`) picks one layer: **factor** (add/replace one `fac_<name>`, max 8), **model**
(rewrite `combine(F, bundle, params)`, e.g. purged walk-forward ridge/HGB) or **portfolio** (SPEC/universe knobs, no
code). `fm.assemble` renders one ordinary spec module, so zoo_val.check, the evaluator and the score are unchanged.
Unlike the qlib bandit, the reward is the score improvement over the best at proposal time (failures = -0.5), and
the bandit state is replayed from the trace, so resumed sessions stay consistent. Launch like zoo_spec:
`bin/zoo_spec_job.sh` style with `rdagent zoo_fm --parent <xs stem> --loop-n N`.

## Acceptance and what the LLM sees (2026-11-27)

Acceptance is zoo-opt's `research_opt/opt_gate.decide()` (`ZOO_GATE_DIR`, default `~/zoo-opt/research_opt`), via
`gate.py`: a candidate must first clear DEV Sharpe >= best + 0.05 (no VAL query spent otherwise); then one VAL query
is charged (`charge_val_query`, max 3 per strategy and `ZOO_VAL_ROUND`, default one round per run; the loop stops
when the budget is used) and `decide()` runs on the sealed record. The LLM gets the DEV-only agent view
(`local_eval.py --dev-only`: no VAL, no gate booleans, no verdict) and the gate's pass/fail with coarse reasons.
VAL numbers live only in `$ZOO_WORK/sealed/` (evaluator work dir and per-spec sealed records, incl. BTC beta).

## Continuous runs with a shared dollar budget

`bin/zoo_continuous.sh <name> <parent> <zoo_fm|zoo_spec>` with `ZOO_USD_BUDGET=50` keeps one strategy's loop running,
resumes the RD-Agent session after crashes and stops when the budget shared by all jobs
(`~/zoo_rd_work/llm_usage_usd.json`, file-locked) is reached. Each call first reserves its worst case
(`ZOO_RESERVE_OUT_TOKENS` output tokens) and is then settled with the provider-reported usage (`CHAT_STREAM=False`
so litellm returns usage; reasoning tokens count as output). Prices: `ZOO_PRICE_IN` / `ZOO_PRICE_OUT` (USD per 1M).
VAL queries: max 3 per strategy per `ZOO_VAL_ROUND_ITERS` (10) iterations; beyond that a DEV-passing candidate is
rejected with `val_budget_round` and the loop keeps exploring. Accepted specs are copied to
`~/zoo_rd_work/accepted/<parent>/`.

## Same-volatility comparison (2026-11-27, agreed by the owner)

Before `opt_gate.decide()`, `gate.same_vol()` scales the candidate's net series by L = parent DEV vol / candidate
DEV vol (VAL keeps the DEV factor). Drawdown (arithmetic cumulative, exact under scaling), BTC beta and annual
return are compared after scaling; Sharpe thresholds are unchanged. Extra rule: L <= 2.0 (`dev_leverage`). No separate
return floor: at equal volatility annual return equals Sharpe x vol, which the Sharpe margins already cover. opt_gate itself is not
changed; RD-Agent passes it already-scaled values. `python -m rdagent.scenarios.zoo.gate STEM PARENT` re-checks an
accepted candidate under the current rules without charging a VAL query.
