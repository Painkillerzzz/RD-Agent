from __future__ import annotations

import re

from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.zoo import evaluation as ev
from rdagent.scenarios.zoo.conf import ZOO_PROP_SETTING as S

RULES = """\
# Task
You improve one trading strategy spec of the quant-strategy-zoo. A spec is a Python module with a SPEC dict and
`signal(bundle, params) -> DataFrame`, backtested by the zoo_val engines (xs = cross-sectional scores, ts = target
positions in [-1, 1]). The market is Binance USDT-M perpetuals, hourly bars.

# Data available to signal() (bundle.wide(field, "1h") gives a ts x symbol frame)
- bars_1h: open, high, low, close, volume, quote_volume, n_trades, taker_buy_volume
- funding (if listed in SPEC["tables"]): funding_rate at settlement times (8h)
- oi (if listed): open_interest, open_interest_value, long_short_ratio, top_trader_ls_ratio; a row stamped ts is only
  known at ts+1h, exactly like a bar (the engines already handle the one-bar convention for bars)
Row t of a bar is known at ts+freq. The engine maps signal_t to the next bar's return; do NOT shift yourself.

# Hard rules (violations fail the evaluator)
- No look-ahead: no shift(-k), no full-sample mean/std/quantile, no centered rolling, no fitting on future rows.
  Walk-forward models must train only on labels that ended before the prediction time (purge the label horizon).
- The output must be NaN where close is NaN (`.where(close.notna())`).
- Keep SPEC keys and contract; params must stay numeric where they are tuning knobs (the evaluator perturbs the
  first six numeric params by +-20% and requires all perturbed runs to stay positive).
- Runtime: a full DEV+VAL evaluation with robustness reruns must finish within ~2 hours on 8 cores.

# Splits and what you may see
DEV = start of data .. 2023-06-30 (you see full DEV metrics). VAL = 2023-07-01 .. 2024-12-31: you only see a bucket
(pass > 0.8 Sharpe / borderline / fail) and gate booleans. TEST/FORWARD are sealed on the server.
Optimise for DEV quality and robustness, not for VAL: VAL is a check. Costs are already netted (Binance taker
cost); the evaluator also reruns at 2x cost.

# Objective (DEV only)
score = 0.5 * DEV Sharpe + 0.5 * min DEV Sharpe over +-20% parameter perturbations, valid only if DEV t >= 2 and VAL
is not 'fail'. Prefer changes with a clear market mechanism over parameter fiddling; simpler is better; lower
turnover and smaller drawdown are good tie-breakers. Fewer, well-motivated changes per iteration.
"""


class ZooSpecScenario(Scenario):
    def __init__(self) -> None:
        self.parent = S.parent
        self.family = ev.family_of(S.parent)
        self.parent_code = ev.parent_code(S.parent)
        row = ev.parent_row(S.parent)
        self.parent_note = row.get("note", "")
        readme = (ev.REPO / "validation" / "README.md").read_text()
        m = re.search(r"## 1\. 数据契约.*?(?=## 4\.)", readme, re.S)
        self.contract = m.group(0) if m else ""

    @property
    def background(self) -> str:
        return RULES

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return "Binance USDT-M perpetual hourly bars, funding and open interest (zoo_val crypto-perp market)."

    @property
    def rich_style_description(self) -> str:
        return f"quant-strategy-zoo spec optimisation, starting from {self.parent} (family {self.family})"

    def get_scenario_all_desc(self, task: Task | None = None, filtered_tag: str | None = None,
                              simple_background: bool | None = None) -> str:
        out = RULES
        if not simple_background:
            out += "\n# zoo_val contract (from validation/README.md)\n" + self.contract
        out += f"\n# Starting spec: {self.parent}\nAuthor's note: {self.parent_note}\n"
        return out

    def get_runtime_environment(self) -> str:
        return "python 3.12, pandas 2.3, numpy, scipy, scikit-learn; no network; 8 cores"
