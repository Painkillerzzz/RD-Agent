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
